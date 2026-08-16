"""Characterization tests for `_sdpa_mask` — the mask-normalisation rules.

`_sdpa_mask` funnels four different user-facing mask spellings into one additive
SDPA mask, and the normalisation has sharp edges that are easy to trip over.
These tests pin the normalisation rules down so they cannot drift silently.

Rank-2 masks are the sharp edge. A `(batch, seq)`-shaped mask matches both a
per-row mask and a `(q_len, kv_len)` attention mask, and the two readings are
exact opposites -- an attention mask *keeps* where it is True, a padding mask
*drops* where it is True. `_sdpa_mask` used to guess, rerouting such a tensor to
the padding channel without inverting it, so `torch.ones(batch, seq)` ("attend
to everything") masked everything, and the ordinary square `(S, S)` causal mask
broke whenever `batch == seq_len`. Both failed silently: no exception, no NaN,
just different logits.

That guess is now a loud `ValueError`. The tests below cover the rejection, the
two documented ways to be explicit, and the unambiguous shapes that still work.
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


@pytest.mark.parametrize("shape", [(2, 4), (2, 6)], ids=["matches_q_len", "matches_kv_len"])
def test_ambiguous_rank_two_mask_is_rejected(shape):
    """A `(batch, seq)`-shaped `attn_mask` must be refused, not guessed at."""
    m = torch.ones(shape, dtype=torch.bool)
    with pytest.raises(ValueError, match="ambiguous rank-2 attn_mask") as excinfo:
        sdpa_mask(attn_mask=m, q_len=4, kv_len=6, cache_len=2, batch=2)
    message = str(excinfo.value)
    assert "key_padding_mask" in message, "the error must name the padding remedy"
    assert "(batch, 1, q_len, kv_len)" in message, "and the attention-mask remedy"


def test_unambiguous_rank_two_mask_is_still_accepted():
    """A `(q_len, kv_len)` mask that cannot be confused with a row mask still works."""
    keep = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    mask, is_causal = sdpa_mask(attn_mask=keep, q_len=4, kv_len=4, batch=2)
    assert is_causal is False
    assert mask[3, 0] == 0.0, "True keeps the key"
    assert mask[0, 3] == float("-inf"), "False drops it"


def test_rank_two_keep_mask_is_rejected_at_model_level(small):
    """End to end: "attend to everything" must raise instead of masking everything."""
    torch.manual_seed(1)
    ids = torch.randint(0, 100, (2, 16))
    keep_all = torch.ones(2, 16, dtype=torch.bool)  # "attend to every key"
    with pytest.raises(ValueError, match="ambiguous rank-2 attn_mask"):
        small(ids, episode_reset=True, attn_mask=keep_all)


def test_documented_remedies_for_a_rank_two_keep_mask(small):
    """Both escapes named in the error message must give the intended result."""
    torch.manual_seed(1)
    ids = torch.randint(0, 100, (2, 16))
    keep_all = torch.ones(2, 16, dtype=torch.bool)
    with torch.no_grad():
        base, _, _, _, _ = small(ids, episode_reset=True)
        as_rank4, _, _, _, _ = small(
            ids, episode_reset=True, attn_mask=keep_all[:, None, None, :].expand(2, 1, 16, 16)
        )
        as_padding, _, _, _, _ = small(
            ids, episode_reset=True, key_padding_mask=~keep_all  # keep -> padding is an inversion
        )
    assert close(base, as_rank4), "keeping every key is a no-op on top of causal"
    assert close(base, as_padding), "and so is declaring nothing to be padding"


def test_square_causal_mask_is_rejected_only_when_batch_equals_seq_len(small):
    """The `(S, S)` causal mask works, except where its shape becomes ambiguous.

    At `batch == seq_len` it also matches `(batch, q_len)`; that used to be read
    as padding, so the same mask was correct at one batch size and wrong at
    another. Now it raises, and rank-4 remains available.
    """
    def setup(batch, seq):
        torch.manual_seed(2)
        ids = torch.randint(0, 100, (batch, seq))
        return ids, torch.tril(torch.ones(seq, seq, dtype=torch.bool))

    ids, tri = setup(2, 16)
    with torch.no_grad():
        base, _, _, _, _ = small(ids, episode_reset=True)
        out, _, _, _, _ = small(ids, episode_reset=True, attn_mask=tri)
    assert close(base, out), "batch != seq_len stays unambiguous and must keep working"

    ids, tri = setup(4, 4)
    with pytest.raises(ValueError, match="ambiguous rank-2 attn_mask"):
        small(ids, episode_reset=True, attn_mask=tri)
    with torch.no_grad():
        base, _, _, _, _ = small(ids, episode_reset=True)
        out, _, _, _, _ = small(ids, episode_reset=True, attn_mask=tri.expand(4, 1, 4, 4))
    assert close(base, out), "the rank-4 spelling is unambiguous and still allowed"


def test_rank_two_mask_keeps_its_meaning_when_a_padding_mask_is_present(small):
    """The ambiguity check only fires when `key_padding_mask` is None.

    Supplying a padding mask already answers the question the check asks, so a
    `(B, KV)` `attn_mask` alongside it is unambiguously keep-semantics,
    broadcast over queries.
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
