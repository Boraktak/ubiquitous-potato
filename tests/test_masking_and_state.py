"""Attention masking, positional offsets, dropout and episode-reset semantics."""

import pytest
import torch

from conftest import close
from dialectic import DialecticTransformer


@pytest.fixture(scope="module")
def small():
    torch.manual_seed(0)
    net = DialecticTransformer(
        vocab_size=100, d_model=64, num_layers=2, num_heads=4,
        state_dim=32, chunk_size=8, mem_slots=4,
    )
    net.eval()
    return net


@pytest.fixture(scope="module")
def small_ids():
    torch.manual_seed(1)
    return torch.randint(0, 100, (2, 16))


def test_explicit_causal_mask_matches_builtin_causal(small, small_ids):
    """A boolean lower-triangular mask must reproduce the default causal path."""
    tri = torch.tril(torch.ones(16, 16, dtype=torch.bool))
    with torch.no_grad():
        base, _, _, _, _ = small(small_ids, episode_reset=True)
        masked, _, _, _, _ = small(small_ids, episode_reset=True, attn_mask=tri)
    assert close(base, masked)


def test_float_additive_mask_matches_bool_mask(small, small_ids):
    """The -inf additive form must be accepted and behave identically."""
    tri = torch.tril(torch.ones(16, 16, dtype=torch.bool))
    additive = torch.zeros(16, 16).masked_fill(~tri, float("-inf"))
    with torch.no_grad():
        base, _, _, _, _ = small(small_ids, episode_reset=True)
        added, _, _, _, _ = small(small_ids, episode_reset=True, attn_mask=additive)
    assert close(base, added)


def test_uniform_pos_offset_is_rope_invariant(small, small_ids):
    """RoPE encodes *relative* position, so shifting every token must be a no-op."""
    with torch.no_grad():
        at_zero, _, _, _, _ = small(small_ids, episode_reset=True, pos_offset=0.0)
        shifted, _, _, _, _ = small(small_ids, episode_reset=True, pos_offset=5.0)
    assert close(at_zero, shifted, rtol=1e-4, atol=1e-4)


def test_per_batch_pos_offset_applies_rowwise(small, small_ids):
    """A tensor offset must shift each batch row independently."""
    with torch.no_grad():
        at_zero, _, _, _, _ = small(small_ids, episode_reset=True, pos_offset=0.0)
        at_five, _, _, _, _ = small(small_ids, episode_reset=True, pos_offset=5.0)
        per_row, _, _, _, _ = small(
            small_ids, episode_reset=True, pos_offset=torch.tensor([0.0, 5.0])
        )
    assert close(per_row[0], at_zero[0])
    assert close(per_row[1], at_five[1])


def test_pos_advances_by_sequence_length(small, small_ids):
    """State position must track how many tokens have been consumed."""
    with torch.no_grad():
        _, states, _, _, _ = small(small_ids, episode_reset=True)
        _, states2, _, _, _ = small(small_ids, states=states, episode_reset=False)
    assert states[0].pos.tolist() == [16.0, 16.0]
    assert states2[0].pos.tolist() == [32.0, 32.0]


def test_episode_reset_matches_a_fresh_forward(small, small_ids):
    """Passing stale state with episode_reset=True must discard it entirely."""
    with torch.no_grad():
        fresh, _, _, _, _ = small(small_ids, episode_reset=True)
        _, stale, _, _, _ = small(small_ids, episode_reset=True)
        after_reset, _, _, _, _ = small(small_ids, states=stale, episode_reset=True)
        carried, _, _, _, _ = small(small_ids, states=stale, episode_reset=False)
    assert close(fresh, after_reset)
    assert not close(fresh, carried), "episode_reset=False should carry state"


def test_non_causal_model_runs(small_ids):
    """causal=False is a supported configuration."""
    torch.manual_seed(0)
    net = DialecticTransformer(
        vocab_size=100, d_model=64, num_layers=2, num_heads=4,
        state_dim=32, chunk_size=8, mem_slots=4, causal=False,
    )
    net.eval()
    with torch.no_grad():
        out, _, _, _, _ = net(small_ids, episode_reset=True)
    assert bool(torch.isfinite(out).all().item())


def test_dropout_active_in_train_and_disabled_in_eval(small, small_ids):
    """Two eval passes must be identical; two train passes must not be."""
    small.train()
    with torch.no_grad():
        torch.manual_seed(3)
        a, _, _, _, _ = small(small_ids, episode_reset=True)
        torch.manual_seed(4)
        b, _, _, _, _ = small(small_ids, episode_reset=True)
    small.eval()
    with torch.no_grad():
        c, _, _, _, _ = small(small_ids, episode_reset=True)
        d, _, _, _, _ = small(small_ids, episode_reset=True)
    assert not torch.equal(a, b), "dropout should randomise training forwards"
    assert torch.equal(c, d), "eval forwards must be deterministic"
