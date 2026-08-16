"""Padding and mask plumbing: 3-D/4-D attention masks and per-row write masks.

Two separate mask channels reach a layer and both end up in `_sdpa_mask`:

* `attn_mask` — 2-D, 3-D `(B, Q, KV)` or 4-D `(B, H, Q, KV)`, bool (True = keep)
  or float (additive, -inf = drop). The 2-D form is already covered by
  `test_masking_and_state.py`; the broadcasting branches for the 3-D and 4-D
  forms are exercised here.
* `key_padding_mask` — `(B, KV)`, True = padding. A layer turns it into the
  SSM's `valid_mask`, which drives two very different code paths: a chunk that
  is padding for *every* row is skipped through a frozen fast path, while a
  chunk that is padding for only *some* rows goes through the normal
  accumulation with a per-row episodic write mask (`_segment_write_mask`).

That second case is why the write-mask tests below all use a *mixed* batch (one
real row, one fully padded row). A uniform batch is useless as a regression
test here: when every row is padding the fast path skips the write entirely, so
a write mask that had been collapsed to a single batch-wide flag would still
look correct. Only a mixed batch can tell "write for the rows that carry real
tokens" apart from "write for everyone".
"""

import pytest
import torch

from conftest import close
from dialectic import ChunkwiseSSM, DialecticTransformer

SEQ = 16
D_MODEL = 32
STATE_DIM = 16
CHUNK = 4
SLOTS = 4


@pytest.fixture(scope="module")
def small() -> DialecticTransformer:
    torch.manual_seed(0)
    net = DialecticTransformer(
        vocab_size=100, d_model=64, num_layers=2, num_heads=4,
        state_dim=32, chunk_size=8, mem_slots=4,
    )
    net.eval()
    return net


@pytest.fixture(scope="module")
def small_ids() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, 100, (2, SEQ))


@pytest.fixture(scope="module")
def baseline(small, small_ids) -> torch.Tensor:
    """Plain causal forward — every mask below must reproduce it."""
    with torch.no_grad():
        logits, _, _, _, _ = small(small_ids, episode_reset=True)
    return logits


@pytest.fixture(scope="module")
def ssm() -> ChunkwiseSSM:
    torch.manual_seed(2)
    net = ChunkwiseSSM(d_model=D_MODEL, state_dim=STATE_DIM, chunk_size=CHUNK, mem_slots=SLOTS)
    net.eval()
    return net


@pytest.fixture(scope="module")
def ssm_input() -> torch.Tensor:
    torch.manual_seed(3)
    return torch.randn(2, 2 * CHUNK, D_MODEL)


def causal() -> torch.Tensor:
    return torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool))


def test_three_dim_bool_mask_matches_builtin_causal(small, small_ids, baseline):
    """A per-batch `(B, Q, KV)` mask must broadcast over heads."""
    mask = causal().unsqueeze(0).expand(2, -1, -1)
    with torch.no_grad():
        out, _, _, _, _ = small(small_ids, episode_reset=True, attn_mask=mask)
    assert close(baseline, out)


def test_three_dim_mask_applies_per_batch_row(small, small_ids, baseline):
    """Rows of a 3-D mask must stay independent — proof the mask is not ignored."""
    mask = causal().unsqueeze(0).repeat(2, 1, 1)
    mask[1, :, 0] = False  # row 1 may not attend to the first token
    mask[1].fill_diagonal_(True)  # ...but never leave a query with zero keys
    with torch.no_grad():
        out, _, _, _, _ = small(small_ids, episode_reset=True, attn_mask=mask)
    assert close(baseline[0], out[0]), "row 0 was not masked and must be untouched"
    assert not close(baseline[1], out[1]), "row 1 lost a key and must change"


def test_four_dim_masks_match_builtin_causal(small, small_ids, baseline):
    """Both `(B, 1, Q, KV)` and `(B, H, Q, KV)` must be accepted as-is."""
    tri = causal().view(1, 1, SEQ, SEQ)
    broadcast = tri.expand(2, 1, -1, -1)
    per_head = tri.expand(2, small.layers[0].num_heads, -1, -1)
    with torch.no_grad():
        out_b, _, _, _, _ = small(small_ids, episode_reset=True, attn_mask=broadcast)
        out_h, _, _, _, _ = small(small_ids, episode_reset=True, attn_mask=per_head)
    assert close(baseline, out_b)
    assert close(baseline, out_h)


def test_four_dim_float_additive_mask_matches_bool_mask(small, small_ids, baseline):
    """The additive -inf form of a 4-D mask must behave like the bool form."""
    additive = torch.zeros(2, 1, SEQ, SEQ).masked_fill(~causal().view(1, 1, SEQ, SEQ), float("-inf"))
    with torch.no_grad():
        out, _, _, _, _ = small(small_ids, episode_reset=True, attn_mask=additive)
    assert close(baseline, out)
    assert bool(torch.isfinite(out).all().item()), "-inf rows must not leak NaN"


def test_fully_padded_chunk_freezes_state_and_ignores_its_content(ssm, ssm_input):
    """A chunk that is padding for every row must be skipped, not accumulated."""
    valid = torch.ones(2, 2 * CHUNK, dtype=torch.bool)
    valid[:, CHUNK:] = False

    noise = ssm_input.clone()
    torch.manual_seed(4)
    noise[:, CHUNK:] = torch.randn(2, CHUNK, D_MODEL)  # different junk under the pad

    with torch.no_grad():
        out, state = ssm(ssm_input, None, True, valid)
        out_noise, state_noise = ssm(noise, None, True, valid)
        out_prefix, state_prefix = ssm(ssm_input[:, :CHUNK], None, True, None)

    assert torch.equal(out, out_noise), "whatever sits under the pad must not matter"
    assert torch.equal(state.hidden, state_noise.hidden)
    assert close(out[:, :CHUNK], out_prefix), "the real prefix is unaffected by the pad"
    assert close(state.hidden, state_prefix.hidden), "the padded chunk must not move the state"
    assert torch.equal(state.mem_valid, state_prefix.mem_valid), "and must not write a slot"
    # Frozen means the readout repeats the held state instead of drifting.
    assert close(out[:, CHUNK:], out[:, CHUNK : CHUNK + 1].expand(-1, CHUNK, -1))


def test_write_mask_is_per_row_in_a_mixed_batch(ssm, ssm_input):
    """Mixed batch: a fully padded row must not write while its neighbour does.

    Both rows go through the normal accumulation path here (the all-padding fast
    path needs *every* row to be padding), so the episodic write is gated only
    by the per-row mask. Collapsing that mask to one batch-wide flag would give
    the padded row a slot full of garbage.
    """
    valid = torch.ones(2, 2 * CHUNK, dtype=torch.bool)
    valid[1] = False  # row 1 is pure padding, row 0 is entirely real

    with torch.no_grad():
        out, state = ssm(ssm_input, None, True, valid)
        solo_out, solo_state = ssm(ssm_input[:1], None, True, valid[:1])

    assert state.mem_valid.sum(dim=-1).tolist() == [2, 0], "only the real row may write"
    assert state.memory[1].abs().max().item() == 0.0, "the padded row's buffer stays empty"
    assert close(out[:1], solo_out), "the padded row must not perturb its neighbour"
    assert close(state.hidden[:1], solo_state.hidden)
    assert close(state.memory[:1], solo_state.memory)


def test_partially_padded_row_writes_fewer_slots(ssm, ssm_input):
    """Per-chunk granularity: a row padded halfway writes one slot, not two."""
    valid = torch.ones(2, 2 * CHUNK, dtype=torch.bool)
    valid[1, CHUNK:] = False  # row 1 runs out of tokens after the first chunk

    with torch.no_grad():
        out, state = ssm(ssm_input, None, True, valid)
        solo_out, solo_state = ssm(ssm_input[1:], None, True, valid[1:])

    assert state.mem_valid.sum(dim=-1).tolist() == [2, 1]
    assert close(out[1:, :CHUNK], solo_out[:, :CHUNK]), "row 1 matches its solo run"
    assert close(state.memory[1:], solo_state.memory)


def test_streaming_with_key_padding_mask_matches_full_forward(small, small_ids):
    """Padding must survive being split across two cached calls.

    The second call only knows about its own 8 tokens, so the layer has to
    extend the padding mask over the cached keys itself.
    """
    pad = torch.zeros(2, SEQ, dtype=torch.bool)
    pad[:, -4:] = True
    half = SEQ // 2
    with torch.no_grad():
        full, _, _, _, _ = small(small_ids, episode_reset=True, key_padding_mask=pad, use_cache=True)
        pre, state, cache, _, _ = small(
            small_ids[:, :half], episode_reset=True, key_padding_mask=pad[:, :half], use_cache=True
        )
        post, _, _, _, _ = small(
            small_ids[:, half:], states=state, caches=cache, episode_reset=False,
            key_padding_mask=pad[:, half:], use_cache=True,
        )
    assert close(full[:, :half], pre)
    assert close(full[:, half:], post)
