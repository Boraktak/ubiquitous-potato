"""`ChunkwiseSSM._run_mixed`: the slow path for batches with ragged chunk phases.

`ChunkwiseSSM.forward` keeps a per-row `phase` (how far the row already is into
its current chunk). When every row shares the same phase it takes the fast,
fully batched `_run_aligned` path. As soon as the phases disagree — which is
exactly what happens with continuous batching, where rows join the batch after
prompts of different lengths — it must fall back to `_run_mixed`, which walks
the batch row by row.

The contract under test is the one that makes that fallback trustworthy:

    running a mixed-phase batch must be *identical* to running each row on its
    own, for the outputs and for every field of the returned state.

A batch whose rows all share one phase can never catch a bug here, because it
never leaves `_run_aligned`.
"""

import pytest
import torch

from dialectic import ChunkwiseSSM, RecurrentState

D_MODEL = 32
STATE_DIM = 16
CHUNK = 4
SLOTS = 4


@pytest.fixture(scope="module")
def ssm() -> ChunkwiseSSM:
    torch.manual_seed(0)
    net = ChunkwiseSSM(d_model=D_MODEL, state_dim=STATE_DIM, chunk_size=CHUNK, mem_slots=SLOTS)
    net.eval()
    return net


def cat_states(*states: RecurrentState) -> RecurrentState:
    """Stack single-row states into one batched state (continuous batching)."""
    return RecurrentState(*[torch.cat(field, dim=0) for field in zip(*states)])


def warmup(ssm: ChunkwiseSSM, length: int, seed: int) -> RecurrentState:
    """Consume `length` tokens on a batch of one, leaving phase = length % CHUNK."""
    torch.manual_seed(seed)
    with torch.no_grad():
        _, state = ssm(torch.randn(1, length, D_MODEL), None, True, None)
    return state


def reset_stats(ssm: ChunkwiseSSM) -> None:
    """`_run_aligned`/`_run_mixed` accumulate into counters `forward` sets up."""
    ssm._stat_n = 0
    ssm._stat_read = 0.0
    ssm._stat_inj = 0.0
    ssm._stat_valid = 0.0


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Row-splitting must be numerically exact, not merely close."""
    return torch.allclose(a, b, rtol=0.0, atol=1e-6)


def assert_rows_match(batched, per_row) -> None:
    """Compare a batched (out, state) result against a list of single-row ones."""
    out, state = batched
    for i, (row_out, row_state) in enumerate(per_row):
        assert same(out[i : i + 1], row_out), f"row {i}: outputs diverge"
        assert same(state.hidden[i : i + 1], row_state.hidden), f"row {i}: hidden diverges"
        assert same(state.memory[i : i + 1], row_state.memory), f"row {i}: memory diverges"
        assert same(state.chunk_drift[i : i + 1], row_state.chunk_drift), f"row {i}: drift diverges"
        assert torch.equal(state.mem_valid[i : i + 1], row_state.mem_valid), f"row {i}: mem_valid"
        assert torch.equal(state.phase[i : i + 1], row_state.phase), f"row {i}: phase"


def test_forward_dispatches_to_mixed_path_only_when_phases_differ(ssm, monkeypatch):
    """Ragged phases must take `_run_mixed`; agreeing phases must stay aligned."""
    calls = {"mixed": 0, "aligned": 0}
    for name in ("_run_mixed", "_run_aligned"):
        original = getattr(ssm, name)
        key = name.replace("_run_", "")

        def spy(*args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(ssm, name, spy)

    aligned_state = cat_states(warmup(ssm, 3, seed=1), warmup(ssm, 7, seed=2))  # phases 3 and 3
    ragged_state = cat_states(warmup(ssm, 3, seed=1), warmup(ssm, 5, seed=2))  # phases 3 and 1
    assert torch.equal(aligned_state.phase, torch.tensor([3, 3]))
    assert torch.equal(ragged_state.phase, torch.tensor([3, 1]))

    torch.manual_seed(10)
    x = torch.randn(2, 6, D_MODEL)
    with torch.no_grad():
        calls["mixed"] = calls["aligned"] = 0
        ssm(x, aligned_state, False, None)
        assert (calls["mixed"], calls["aligned"]) == (0, 1), "equal phases must use the fast path"

        calls["mixed"] = calls["aligned"] = 0
        ssm(x, ragged_state, False, None)
        assert calls["mixed"] == 1, "ragged phases must use the slow path"
        assert calls["aligned"] == 2, "the slow path runs one aligned pass per row"


def test_mixed_batch_matches_per_row_processing(ssm):
    """Core contract: a ragged batch equals the rows processed separately."""
    row0, row1 = warmup(ssm, 3, seed=1), warmup(ssm, 5, seed=2)
    torch.manual_seed(11)
    x = torch.randn(2, 7, D_MODEL)

    with torch.no_grad():
        batched = ssm(x, cat_states(row0, row1), False, None)
        per_row = [ssm(x[i : i + 1], st, False, None) for i, st in enumerate((row0, row1))]

    assert_rows_match(batched, per_row)
    # The rows really are out of step: they close their chunks at different
    # token counts, so a bug that shares one phase across the batch would show.
    assert not torch.equal(batched[1].phase[0], batched[1].phase[1])


def test_mixed_batch_with_padding_matches_per_row_processing(ssm):
    """The ragged path must respect a per-row validity mask as well."""
    row0, row1 = warmup(ssm, 2, seed=3), warmup(ssm, 7, seed=4)
    torch.manual_seed(12)
    x = torch.randn(2, 9, D_MODEL)
    valid = torch.ones(2, 9, dtype=torch.bool)
    valid[0, 6:] = False  # row 0 ends early
    valid[1, 4:] = False  # row 1 ends even earlier

    with torch.no_grad():
        batched = ssm(x, cat_states(row0, row1), False, valid)
        per_row = [
            ssm(x[i : i + 1], st, False, valid[i : i + 1]) for i, st in enumerate((row0, row1))
        ]

    assert_rows_match(batched, per_row)


def test_three_distinct_phases_match_per_row_processing(ssm):
    """More than two phases must not confuse the row-by-row bookkeeping."""
    rows = [warmup(ssm, n, seed=20 + n) for n in (1, 2, 3)]
    assert [int(r.phase.item()) for r in rows] == [1, 2, 3]
    torch.manual_seed(13)
    x = torch.randn(3, 5, D_MODEL)

    with torch.no_grad():
        batched = ssm(x, cat_states(*rows), False, None)
        per_row = [ssm(x[i : i + 1], st, False, None) for i, st in enumerate(rows)]

    assert_rows_match(batched, per_row)
    assert batched[1].mem_valid.sum(dim=-1).tolist() == [1, 1, 2], "rows write at their own pace"


def test_mixed_path_equals_aligned_path_when_phases_agree(ssm):
    """The two implementations must agree wherever their domains overlap."""
    torch.manual_seed(14)
    x = torch.randn(2, 10, D_MODEL)
    state = RecurrentState.zeros(2, STATE_DIM, SLOTS, x.device, x.dtype)
    u = ssm.input_proj(x)

    with torch.no_grad():
        reset_stats(ssm)
        aligned = ssm._run_aligned(u, state.hidden, state.memory, state.mem_valid, 0, state.chunk_drift, None)
        opens_aligned = ssm._stat_n
        reset_stats(ssm)
        mixed = ssm._run_mixed(u, state.hidden, state.memory, state.mem_valid, state.phase, state.chunk_drift, None)
        opens_mixed = ssm._stat_n

    out_a, st_a, mem_a, valid_a, phase_a, drift_a = aligned
    out_m, st_m, mem_m, valid_m, phase_m, drift_m = mixed
    assert same(out_a, out_m)
    assert same(st_a, st_m)
    assert same(mem_a, mem_m)
    assert same(drift_a, drift_m)
    assert torch.equal(valid_a, valid_m)
    assert torch.equal(phase_m, torch.full((2,), phase_a, dtype=phase_m.dtype))
    assert opens_aligned > 0 and opens_mixed == opens_aligned * 2, "one open per row per chunk"


def test_mixed_path_handles_empty_sequence(ssm):
    """S=0 must return an empty output and hand the state straight back."""
    state = RecurrentState.zeros(2, STATE_DIM, SLOTS, torch.device("cpu"), torch.float32)
    phase = torch.tensor([1, 3])
    empty = torch.zeros(2, 0, STATE_DIM)

    with torch.no_grad():
        reset_stats(ssm)
        out, hidden, memory, mem_valid, phase_out, drift = ssm._run_mixed(
            empty, state.hidden, state.memory, state.mem_valid, phase, state.chunk_drift, None
        )

    assert out.shape == (2, 0, D_MODEL)
    assert same(hidden, state.hidden)
    assert same(memory, state.memory)
    assert same(drift, state.chunk_drift)
    assert torch.equal(mem_valid, state.mem_valid)
    assert torch.equal(phase_out, phase), "an empty step must not advance the phase"
