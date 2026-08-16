"""Chunk-invariance, masking and generation consistency checks."""

import torch

from conftest import VOCAB, close


def test_empty_sequence(model, ids):
    """S=0 must return an empty logits tensor instead of crashing."""
    with torch.no_grad():
        logits, _, _, _, _ = model(ids[:, :0], episode_reset=True)
    assert logits.shape == (ids.shape[0], 0, VOCAB)


def test_ignore_index_masking(model, ids, targets):
    """Masked-out target positions must still yield a finite loss."""
    masked = targets.clone()
    masked[:, :8] = -100
    with torch.no_grad():
        _, _, _, loss, _ = model(ids, targets=masked, episode_reset=True)
    assert loss is not None and torch.isfinite(loss)


def test_generate_shape(model, ids):
    """Greedy decode must append exactly max_new_tokens ids."""
    with torch.no_grad():
        out = model.generate(ids[:, :8], max_new_tokens=4, temperature=0.0)
    assert out.shape == (ids.shape[0], 12)
    assert torch.equal(out[:, :8], ids[:, :8])


def test_memory_carries_across_calls(model, ids, trained_states):
    """A second pass must keep writing slots, not reset them."""
    with torch.no_grad():
        _, states2, _, _, _ = model(ids, states=trained_states, episode_reset=False)
    assert all(bool(s.mem_valid.any().item()) for s in trained_states)
    assert all(not torch.equal(a.memory, b.memory) for a, b in zip(trained_states, states2))


def test_ssm_branch_is_chunk_invariant_without_cache(model, ids):
    """The true chunk-invariant path: the SSM alone needs no KV cache.

    Splitting the sequence at a non-chunk boundary must reproduce the full
    forward exactly, because the SSM carries its recurrent state forward.
    """
    ssm = model.layers[0].ssm
    with torch.no_grad():
        x = model.layers[0].norm1(model.embed(ids))
        full, _ = ssm(x, None, True, None)
        pre, state = ssm(x[:, :8], None, True, None)
        post, _ = ssm(x[:, 8:], state, False, None)
    assert close(full[:, :8], pre)
    assert close(full[:, 8:], post)


def test_full_model_split_mid_chunk_with_cache(model, ids):
    """Full-model invariance needs the KV cache so attention sees all history."""
    with torch.no_grad():
        full, _, _, _, _ = model(ids, episode_reset=True)
        pre, state, cache, _, _ = model(ids[:, :8], episode_reset=True, use_cache=True)
        post, _, _, _, _ = model(
            ids[:, 8:], states=state, caches=cache, episode_reset=False, use_cache=True
        )
    assert close(full[:, :8], pre)
    assert close(full[:, 8:], post)


def test_full_model_split_on_chunk_boundary_with_cache(model, ids):
    """Same guarantee when the split lands exactly on a chunk boundary."""
    with torch.no_grad():
        full, _, _, _, _ = model(ids, episode_reset=True)
        pre, state, cache, _, _ = model(ids[:, :32], episode_reset=True, use_cache=True)
        post, _, _, _, _ = model(
            ids[:, 32:], states=state, caches=cache, episode_reset=False, use_cache=True
        )
    assert close(full[:, :32], pre)
    assert close(full[:, 32:], post)


def test_cached_forward_matches_cached_split(model, ids):
    """use_cache=True on both sides must agree end to end."""
    with torch.no_grad():
        full, _, _, _, _ = model(ids, episode_reset=True, use_cache=True)
        pre, state, cache, _, _ = model(ids[:, :8], episode_reset=True, use_cache=True)
        post, _, _, _, _ = model(
            ids[:, 8:], states=state, caches=cache, episode_reset=False, use_cache=True
        )
    assert close(full[:, :8], pre)
    assert close(full[:, 8:], post)


def test_padding_id_independence(model, ids):
    """Whatever id sits under a padding mask must not change the real outputs."""
    pad = torch.zeros(ids.shape, dtype=torch.bool, device=ids.device)
    pad[:, -4:] = True
    ids_a, ids_b = ids.clone(), ids.clone()
    ids_a[:, -4:] = 0
    ids_b[:, -4:] = 1
    with torch.no_grad():
        logits_a, state_a, _, _, _ = model(ids_a, episode_reset=True, key_padding_mask=pad)
        logits_b, state_b, _, _, _ = model(ids_b, episode_reset=True, key_padding_mask=pad)
    assert close(logits_a[:, :-4], logits_b[:, :-4])
    assert close(state_a[0].hidden, state_b[0].hidden)
    assert bool(torch.isfinite(logits_a).all().item())


def test_fifo_eviction_fills_every_slot(model):
    """A sequence long enough to overflow the buffer must leave all slots valid."""
    length = model.chunk_size * (model.mem_slots + 4)
    torch.manual_seed(7)
    long_ids = torch.randint(0, VOCAB, (2, length))
    with torch.no_grad():
        _, states, _, _, _ = model(long_ids, episode_reset=True)
    assert all(bool(s.mem_valid.all().item()) for s in states)
