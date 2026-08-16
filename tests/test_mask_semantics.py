"""Characterization tests for `_sdpa_mask` — the mask-normalisation rules.

`_sdpa_mask` funnels four different user-facing mask spellings into one additive
SDPA mask, and the normalisation has sharp edges that are easy to trip over.
These tests pin the **current behaviour** down so it cannot drift silently.

Two of them document genuine traps rather than desirable behaviour; they are
marked in the docstrings and mirrored in the "Mask semantics" section of the
README. They are written so that changing the behaviour on purpose fails the
test loudly (and the fix is then a one-line edit here plus a README update),
rather than the behaviour changing under a suite that never noticed.

Nothing here modifies `dialectic.py`.
"""

import pytest
import torch

from conftest import close
from dialectic import DialecticTransformer, TemporalRoPE, _sdpa_mask

DEVICE = torch.device("cpu")
DTYPE = torch.float32


def sdpa_mask(attn_mask=None, key_padding_mask=None, q_len=4, kv_len=4,
              cache_len=0, batch=2, causal=True):
    return _sdpa_mask(attn_mask, key_padding_mask, q_len, kv_len, cache_len,
                      batch, DEVICE, DTYPE, causal)


@pytest.fixture(scope="module")
def small() -> DialecticTransformer:
    torch.manual_seed(0)
    net = DialecticTransformer(
        vocab_size=100, d_model=64, num_layers=2, num_heads=4,
        state_dim=32, chunk_size=8, mem_slots=4,
    )
    net.eval()
    return net


def test_no_mask_uses_the_builtin_causal_fast_path():
    """A plain causal forward must hand SDPA `is_causal=True`, not a materialised mask."""
    mask, is_causal = sdpa_mask()
    assert mask is None and is_causal is True

    mask, is_causal = sdpa_mask(causal=False)
    assert mask is None and is_causal is False, "non-causal without masks is unrestricted"


def test_single_token_decode_with_cache_needs_no_mask():
    """One query against N cached keys may see all of them: no mask, no causal flag."""
    mask, is_causal = sdpa_mask(q_len=1, kv_len=5, cache_len=4)
    assert mask is None
    assert is_causal is False, "is_causal would wrongly re-apply a triangle to a 1-row query"


def test_rank_two_batch_shaped_mask_is_reinterpreted_as_padding():
    """TRAP: a `(B, S)` `attn_mask` is silently re-read as a `key_padding_mask`.

    `attn_mask` is keep-semantics (True = attend) while `key_padding_mask` is
    drop-semantics (True = padding). The re-read does **not** invert the tensor,
    so a `(B, S)` mask means the exact opposite of what its name implies.
    """
    m = torch.tensor([[True, True, False, False], [True, False, False, False]])
    as_attn, _ = sdpa_mask(attn_mask=m)
    as_padding, _ = sdpa_mask(key_padding_mask=m)
    assert torch.equal(as_attn, as_padding), "the (B, S) form is routed to the padding channel"
    # Read at query 3, which the causal triangle lets see every key: the two
    # True entries became -inf, i.e. "keep" was applied as "drop".
    assert as_attn[0, 0, 3, 0] == float("-inf"), "True was dropped, not kept"
    assert as_attn[0, 0, 3, 1] == float("-inf")
    assert as_attn[0, 0, 3, 2] == 0.0, "False was kept, not dropped"


def test_rank_two_keep_mask_flips_meaning_at_model_level(small):
    """TRAP, observable end to end: "attend to everything" silently masks everything.

    No exception and no NaN — just quietly different logits, which is the worst
    possible failure mode. Pass a 3-D/4-D mask (or `key_padding_mask`) instead.
    """
    torch.manual_seed(1)
    ids = torch.randint(0, 100, (2, 16))
    keep_all = torch.ones(2, 16, dtype=torch.bool)  # "attend to every key"
    with torch.no_grad():
        base, _, _, _, _ = small(ids, episode_reset=True)
        out, _, _, _, _ = small(ids, episode_reset=True, attn_mask=keep_all)
    assert not close(base, out), "an all-keep mask is not a no-op here"
    assert bool(torch.isfinite(out).all().item()), "it fails silently rather than loudly"


def test_square_mask_is_reinterpreted_when_batch_equals_seq_len(small):
    """TRAP: the ordinary `(S, S)` causal mask breaks when `batch == seq_len`.

    `(S, S)` then also matches `(batch, q_len)`, and the padding branch wins.
    The same mask on the same model is correct at one batch size and wrong at
    another, which makes this exceptionally easy to miss.
    """
    def run(batch, seq):
        torch.manual_seed(2)
        ids = torch.randint(0, 100, (batch, seq))
        tri = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        with torch.no_grad():
            base, _, _, _, _ = small(ids, episode_reset=True)
            out, _, _, _, _ = small(ids, episode_reset=True, attn_mask=tri)
        return close(base, out)

    assert run(2, 16), "batch != seq_len: the square mask is read as an attention mask"
    assert not run(4, 4), "batch == seq_len: the same mask is read as padding"


def test_rank_two_mask_keeps_its_meaning_when_a_padding_mask_is_present(small):
    """The re-read only fires when `key_padding_mask` is None.

    Supply both and the `(B, KV)` mask stays keep-semantics, broadcast over
    queries — so the very same tensor means opposite things depending on
    whether a padding mask travels with it.
    """
    torch.manual_seed(3)
    ids = torch.randint(0, 100, (2, 16))
    tri = torch.tril(torch.ones(16, 16, dtype=torch.bool))
    keep = torch.ones(2, 16, dtype=torch.bool)
    keep[:, 10:] = False
    no_padding = torch.zeros(2, 16, dtype=torch.bool)
    rank3 = (keep[:, None, :] & tri[None]).clone()

    with torch.no_grad():
        rank2_out, _, _, _, _ = small(
            ids, episode_reset=True, attn_mask=keep, key_padding_mask=no_padding
        )
        rank3_out, _, _, _, _ = small(
            ids, episode_reset=True, attn_mask=rank3, key_padding_mask=no_padding
        )
    assert close(rank2_out, rank3_out), "with a padding mask present, (B, KV) means keep"

    mask, _ = sdpa_mask(attn_mask=keep[:, :4], key_padding_mask=no_padding[:, :4])
    assert mask.shape == (2, 1, 4, 4), "broadcast over heads and queries"


def test_short_padding_mask_is_extended_over_cached_keys():
    """A padding mask covering only the new tokens is zero-padded across the cache."""
    pad = torch.tensor([[False, True], [True, False]])
    mask, is_causal = sdpa_mask(key_padding_mask=pad, q_len=2, kv_len=6, cache_len=4)
    assert is_causal is False
    assert mask.shape == (2, 1, 2, 6)
    assert bool((mask[..., :4] == 0.0).all().item()), "cached keys are assumed non-padding"
    assert mask[0, 0, 0, 5] == float("-inf"), "row 0 pads its second new token"
    assert mask[1, 0, 0, 4] == float("-inf"), "row 1 pads its first new token"


def test_rope_broadcasts_shared_one_dimensional_positions():
    """A `(S,)` position vector must apply to every row of the batch."""
    torch.manual_seed(4)
    rope = TemporalRoPE(8)
    x = torch.randn(2, 3, 4, 8)
    shared = torch.arange(4, dtype=torch.float32)
    per_row = shared.unsqueeze(0).expand(2, -1)
    assert torch.allclose(rope(x, shared), rope(x, per_row))
