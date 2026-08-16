"""Gradient flow checks: every trainable path must receive finite, non-zero grads."""

import pytest
import torch

from conftest import grad_ok


@pytest.fixture(scope="module")
def backward_model(model, ids, targets):
    """Run one training-mode forward/backward and hand back the populated grads."""
    model.train()
    model.zero_grad(set_to_none=True)
    _, states, caches, loss, metrics = model(ids, targets=targets, episode_reset=True)
    assert loss is not None
    loss.backward()
    yield model, states, caches, loss, metrics
    model.zero_grad(set_to_none=True)
    model.eval()


def test_loss_is_finite(backward_model):
    _, _, _, loss, metrics = backward_model
    assert torch.isfinite(loss).all()
    assert metrics["Loss/CE"] == pytest.approx(metrics["Loss/Total"])
    assert loss.item() > 0.0


@pytest.mark.parametrize(
    "path",
    [
        "ssm.f_drift",
        "ssm.episodic.out_proj",
        "ssm.episodic.inject",
        "ssm.episodic.query_proj",
        "ssm.episodic.key_proj",
        "ssm.episodic.val_proj",
    ],
)
def test_gradient_reaches_every_layer(backward_model, path):
    """SSM drift, memory readout, injection and Q/K/V must all be trainable."""
    net = backward_model[0]
    for i, layer in enumerate(net.layers):
        module = layer
        for part in path.split("."):
            module = getattr(module, part)
        assert grad_ok(module.weight), f"no gradient at layer {i} -> {path}"


def test_no_kv_cache_during_training(backward_model):
    """Training forward must not build a KV cache (memory blow-up guard)."""
    assert backward_model[2] is None


def test_state_carry_allows_second_backward(model, ids, targets, backward_model):
    """Carrying detached state across steps must not trip 'backward twice'."""
    states = backward_model[1]
    model.zero_grad(set_to_none=True)
    _, _, _, loss2, _ = model(ids, targets=targets, states=states, episode_reset=False)
    assert loss2 is not None
    loss2.backward()
    assert torch.isfinite(loss2).all()
    model.zero_grad(set_to_none=True)
