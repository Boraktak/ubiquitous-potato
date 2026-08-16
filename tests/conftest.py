"""Shared fixtures for the DialecticTransformer test suite.

All tests run on CPU with a fixed seed so results are deterministic and
reproducible in CI (see .github/workflows/ci.yml).
"""

import pytest
import torch

from dialectic import DialecticTransformer

SEED = 42
VOCAB = 1000
BATCH = 2
SEQ = 256


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(scope="session")
def model(device: torch.device) -> DialecticTransformer:
    """A single deterministic model shared by the whole session.

    Building the model once keeps the suite fast; every test that mutates
    grads calls `zero_grad` and tests never train, so no state leaks.
    """
    torch.manual_seed(SEED)
    net = DialecticTransformer(
        vocab_size=VOCAB,
        d_model=256,
        num_layers=4,
        num_heads=4,
        state_dim=128,
        chunk_size=32,
        mem_slots=16,
    ).to(device)
    net.eval()
    return net


@pytest.fixture(scope="session")
def ids(device: torch.device) -> torch.Tensor:
    torch.manual_seed(SEED + 1)
    return torch.randint(0, VOCAB, (BATCH, SEQ), device=device)


@pytest.fixture(scope="session")
def targets(device: torch.device) -> torch.Tensor:
    torch.manual_seed(SEED + 2)
    return torch.randint(0, VOCAB, (BATCH, SEQ), device=device)


@pytest.fixture(scope="session")
def trained_states(model: DialecticTransformer, ids: torch.Tensor, targets: torch.Tensor):
    """States produced by one full forward pass, used by carry/ablation tests."""
    with torch.no_grad():
        _, states, _, _, _ = model(ids, targets=targets, episode_reset=True)
    return states


def close(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-4) -> bool:
    return torch.allclose(a, b, rtol=rtol, atol=atol)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def grad_ok(p: torch.Tensor) -> bool:
    g = p.grad
    return g is not None and bool(torch.isfinite(g).all().item()) and g.abs().sum().item() > 0.0
