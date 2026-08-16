"""Decoding paths: greedy, temperature sampling and top-k.

`run_demo()` only ever exercised greedy decode (temperature=0.0), so the whole
sampling branch of `generate` was untested.
"""

import torch

from conftest import VOCAB


def test_greedy_matches_argmax_of_full_forward(model, ids):
    """Greedy decode must pick exactly the argmax of a plain forward pass."""
    prompt = ids[:, :8]
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=1, temperature=0.0)
        logits, _, _, _, _ = model(prompt, episode_reset=True)
    assert torch.equal(out[:, -1], logits[:, -1].argmax(dim=-1))


def test_top_k_1_is_equivalent_to_greedy(model, ids):
    """top_k=1 leaves a single candidate, so it must reduce to greedy."""
    prompt = ids[:, :8]
    with torch.no_grad():
        greedy = model.generate(prompt, max_new_tokens=3, temperature=0.0)
        topk1 = model.generate(prompt, max_new_tokens=3, temperature=1.0, top_k=1)
    assert torch.equal(greedy, topk1)


def test_sampling_is_reproducible_under_a_fixed_seed(model, ids):
    """Stochastic decode must still be deterministic given the same seed."""
    prompt = ids[:, :8]
    with torch.no_grad():
        torch.manual_seed(1234)
        first = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=5)
        torch.manual_seed(1234)
        second = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=5)
    assert torch.equal(first, second)
    assert first.shape == (prompt.shape[0], prompt.shape[1] + 5)


def test_sampled_tokens_stay_in_vocab(model, ids):
    """top-k masking must never let -inf positions be sampled."""
    with torch.no_grad():
        torch.manual_seed(7)
        out = model.generate(ids[:, :8], max_new_tokens=6, temperature=0.8, top_k=10)
    assert int(out.min()) >= 0
    assert int(out.max()) < VOCAB


def test_top_k_larger_than_vocab_is_clamped(model, ids):
    """top_k above vocab size must clamp instead of crashing in topk()."""
    with torch.no_grad():
        torch.manual_seed(7)
        out = model.generate(ids[:, :4], max_new_tokens=2, temperature=1.0, top_k=VOCAB * 10)
    assert out.shape == (ids.shape[0], 6)
    assert int(out.max()) < VOCAB


def test_generate_restores_training_mode(model, ids):
    """generate() flips to eval internally; it must put the model back."""
    model.train()
    try:
        with torch.no_grad():
            model.generate(ids[:, :4], max_new_tokens=2, temperature=0.0)
        assert model.training, "generate() left the model in eval mode"
    finally:
        model.eval()


def test_generate_leaves_no_grad_history(model, ids):
    """Decoding is inference-only and must not build a graph."""
    out = model.generate(ids[:, :4], max_new_tokens=2, temperature=0.0)
    assert not out.requires_grad
