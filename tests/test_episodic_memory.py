"""Ablations proving the episodic memory actually influences the output."""

import torch

from conftest import VOCAB, max_abs


def test_retrieve_returns_zero_on_empty_buffer(model, trained_states):
    """No valid slots must read out exactly zero, not NaN from a masked softmax."""
    episodic = model.layers[0].ssm.episodic
    state = trained_states[0]
    live = episodic.retrieve(state.hidden, state.memory, state.mem_valid)
    dead = episodic.retrieve(
        state.hidden, torch.zeros_like(state.memory), torch.zeros_like(state.mem_valid)
    )
    assert live.norm().item() > 0.0
    assert dead.abs().max().item() == 0.0
    assert bool(torch.isfinite(dead).all().item())


def test_zeroing_memory_changes_output(model, trained_states):
    """Ablation: wiping the buffer must measurably move the logits."""
    torch.manual_seed(11)
    probe = torch.randint(0, VOCAB, (2, 64))
    zeroed = [
        s._replace(memory=torch.zeros_like(s.memory), mem_valid=torch.zeros_like(s.mem_valid))
        for s in trained_states
    ]
    with torch.no_grad():
        live, _, _, _, _ = model(probe, states=trained_states, episode_reset=False)
        dead, _, _, _, _ = model(probe, states=zeroed, episode_reset=False)
    assert max_abs(live, dead) > 1e-6


def test_shuffling_memory_across_batch_changes_output(model, trained_states):
    """Ablation: memory must be batch-specific, not an inert constant."""
    torch.manual_seed(11)
    probe = torch.randint(0, VOCAB, (2, 64))
    shuffled = [
        s._replace(memory=s.memory.roll(1, dims=0), mem_valid=s.mem_valid.roll(1, dims=0))
        for s in trained_states
    ]
    with torch.no_grad():
        live, _, _, _, _ = model(probe, states=trained_states, episode_reset=False)
        shuf, _, _, _, _ = model(probe, states=shuffled, episode_reset=False)
    assert max_abs(live, shuf) > 1e-6


def test_write_is_detached(model, trained_states):
    """Stored slots are detached on purpose (truncated BPTT)."""
    episodic = model.layers[0].ssm.episodic
    state = trained_states[0]
    live = state.hidden.clone().requires_grad_(True)
    buf, valid = episodic.write(live, state.memory, state.mem_valid)
    assert not buf.requires_grad
    assert bool(valid[:, -1].all().item())


def test_memory_metrics_are_reported(model, ids, targets):
    """Every layer must publish its Mem/* diagnostics."""
    with torch.no_grad():
        _, _, _, _, metrics = model(ids, targets=targets, episode_reset=True)
    for i in range(len(model.layers)):
        assert metrics[f"Mem/L{i}_opens"] > 0.0
        assert f"Mem/L{i}_read_norm" in metrics
        assert f"Mem/L{i}_inject_norm" in metrics
        assert metrics[f"Mem/L{i}_valid_slots"] >= 0.0
