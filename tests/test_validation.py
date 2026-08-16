"""Constructor and argument validation: bad input must fail loudly and early."""

import pytest
import torch

from dialectic import (
    ChunkwiseSSM,
    DialecticTransformer,
    EpisodicMemory,
    HybridDialecticLayer,
    TemporalRoPE,
)


def test_rope_rejects_odd_dim():
    with pytest.raises(ValueError, match="RoPE dim must be even"):
        TemporalRoPE(7)


def test_episodic_memory_rejects_zero_slots():
    with pytest.raises(ValueError, match="num_slots must be >= 1"):
        EpisodicMemory(32, 0)


def test_ssm_rejects_zero_chunk_size():
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        ChunkwiseSSM(64, 32, 0, 4)


def test_layer_rejects_indivisible_head_count():
    with pytest.raises(ValueError, match="must be divisible by num_heads"):
        HybridDialecticLayer(64, 5)


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


def test_wrong_number_of_states_is_rejected(small, small_ids):
    """A states list that does not match layer count is a silent-corruption risk."""
    with pytest.raises(ValueError, match="states has 1 entries, expected 2"):
        small(small_ids, states=[None])


def test_wrong_number_of_caches_is_rejected(small, small_ids):
    with pytest.raises(ValueError, match="caches has 1 entries, expected 2"):
        small(small_ids, caches=[None])


def test_mismatched_key_padding_mask_is_rejected(small, small_ids):
    with pytest.raises(ValueError, match="key_padding_mask last dim"):
        small(small_ids, episode_reset=True, key_padding_mask=torch.zeros(2, 99, dtype=torch.bool))


def test_weight_tying_is_configurable():
    """Tied by default; untied must allocate a separate head matrix."""
    tied = DialecticTransformer(vocab_size=100, d_model=64, num_layers=1, num_heads=4, state_dim=32)
    untied = DialecticTransformer(
        vocab_size=100, d_model=64, num_layers=1, num_heads=4, state_dim=32, tie_weights=False
    )
    assert tied.lm_head.weight is tied.embed.weight
    assert untied.lm_head.weight is not untied.embed.weight


def test_batch_size_change_reinitialises_state(small, small_ids):
    """Carrying state into a smaller batch must re-init rather than crash."""
    with torch.no_grad():
        _, states, _, _, _ = small(small_ids, episode_reset=True)
        out, _, _, _, _ = small(small_ids[:1], states=states, episode_reset=False)
    assert out.shape[0] == 1
    assert bool(torch.isfinite(out).all().item())
