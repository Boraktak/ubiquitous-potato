# ubiquitous-potato — DialecticTransformer

[![CI](https://github.com/Boraktak/ubiquitous-potato/actions/workflows/ci.yml/badge.svg)](https://github.com/Boraktak/ubiquitous-potato/actions/workflows/ci.yml)

A single-file hybrid sequence model (`dialectic.py`, no dependencies beyond
PyTorch) that runs three branches side by side in every layer:

* a **chunkwise SSM** — a linear-recurrent state that is *chunk-invariant*: you
  can cut a sequence anywhere, carry the state, and get the same numbers back;
* an **episodic memory** — a FIFO of content-addressed slots the SSM reads at
  the start of every chunk and writes at the end of it;
* **causal attention** with RoPE and an optional KV cache, blended into the SSM
  output by a learned **fusion gate**.

The point of the combination: the SSM plus the memory carry unbounded history in
O(1) state, attention supplies exact local recall, and the gate learns how much
of each to trust per token and per channel.

---

## Architecture

```
ids ──► embed ──►┌─────────────── HybridDialecticLayer (xN) ───────────────┐
                 │  x_norm = norm1(x)                                      │
                 │        ├── ChunkwiseSSM(x_norm, state) ──► ssm_out      │
                 │        │       └── EpisodicMemory (read/write)          │
                 │        └── RoPE + causal SDPA(+KV cache) ──► attn       │
                 │  gate  = sigmoid(W[ssm_out ; attn])                     │
                 │  x     = x + gate * ssm_out + (1 - gate) * attn         │
                 │  x     = x + ffn(norm2(x))                              │
                 └─────────────────────────────────────────────────────────┘
                                        └──► norm ──► lm_head ──► logits
```

### Chunkwise SSM

The sequence is cut into fixed `chunk_size` blocks. Inside a block the update is
a plain cumulative sum, which is associative and therefore evaluated in
parallel over the whole block instead of token by token:

```
inc_t   = drift + softplus(input_inject) * (W_in · x_t)
raw_t   = state_0 + Σ_{i<=t} inc_i                 # one cumsum per chunk
gated_t = tanh(LN(raw_t) * sigmoid(W_f · LN(raw_t)))
out_t   = W_out · gated_t
```

Chunk boundaries are where the interesting work happens:

| Event | What the SSM does |
|---|---|
| **chunk open** | read episodic memory with the current state as the query, add the (small, near-identity-initialised) injection to the state, then compute `drift = f_drift(LN(state)) / sqrt(chunk_size)` |
| **chunk close** | write the final gated state into the FIFO (detached), reset `drift`, set `phase = 0` |
| **partial tail** | keep the raw accumulator as the state and remember `phase` so the next call resumes mid-chunk |

`RecurrentState` is what gets carried between calls: `hidden`, `memory`,
`mem_valid`, `pos`, `phase`, `chunk_drift`. Because `phase` is **per row**, rows
of one batch may sit at different offsets inside their chunk (continuous
batching). When they agree the SSM takes the fast batched path
(`_run_aligned`); when they disagree it falls back to `_run_mixed`, which walks
the batch row by row and must produce bit-identical results — see
`tests/test_mixed_phase.py`.

### Episodic memory

`num_slots` FIFO slots per row. Reading is a one-query attention over the valid
slots (an empty buffer reads out exactly zero rather than a NaN from a fully
masked softmax); writing evicts the oldest slot. Writes are **detached on
purpose** — truncated BPTT, no gradient credit flowing through stored slots.
Each layer publishes diagnostics in the metrics dict: `Mem/L{i}_opens`,
`Mem/L{i}_read_norm`, `Mem/L{i}_inject_norm`, `Mem/L{i}_valid_slots`.

### Fusion gate

`fusion_gate` is a `Linear(2 * d_model, d_model)` over `[ssm_out ; attn]` whose
bias starts at `-1.0`, so the gate opens at ≈0.27: attention dominates early in
training and the model has to learn to lean on the recurrent branch.

---

## Usage

### Forward pass and loss

```python
import torch
from dialectic import DialecticTransformer

model = DialecticTransformer(
    vocab_size=1000, d_model=256, num_layers=4, num_heads=4,
    state_dim=128, chunk_size=32, mem_slots=16,
)

ids = torch.randint(0, 1000, (2, 256))
targets = torch.randint(0, 1000, (2, 256))

logits, states, caches, loss, metrics = model(ids, targets=targets, episode_reset=True)

print(logits.shape)              # torch.Size([2, 256, 1000])
print(caches is None)            # True  -> no KV cache is built for training
print(round(metrics["Mem/L0_opens"]))   # 8  -> chunk opens, 256 tokens / chunk_size 32
loss.backward()
```

`forward` always returns the same 5-tuple
`(logits, states, caches, loss, metrics)`; `loss` is `None` when `targets` is
omitted and `caches` is `None` unless `use_cache=True`.

| Argument | Meaning |
|---|---|
| `targets` | labels for cross-entropy; positions equal to `ignore_index` (default `-100`) are skipped |
| `states` | list of one `RecurrentState` per layer, from a previous call |
| `caches` | list of one `LayerCache` per layer, from a previous `use_cache=True` call |
| `episode_reset` | drop the incoming state/cache and start a fresh episode |
| `attn_mask` | `(Q, KV)`, `(B, Q, KV)` or `(B, H, Q, KV)`; bool means *keep*, float is additive |
| `key_padding_mask` | `(B, S)`, `True` marks padding; also drives the SSM's validity mask |
| `pos_offset` | float or `(B,)` tensor added to the RoPE positions |
| `use_cache` | return KV caches for streaming |

### Generation

```python
import torch
from dialectic import DialecticTransformer

torch.manual_seed(0)
model = DialecticTransformer(vocab_size=100, d_model=64, num_layers=2,
                             num_heads=4, state_dim=32, chunk_size=8, mem_slots=4)
prompt = torch.randint(0, 100, (2, 8))

greedy = model.generate(prompt, max_new_tokens=16, temperature=0.0)
sampled = model.generate(prompt, max_new_tokens=16, temperature=0.9, top_k=20)

print(greedy.shape)                              # torch.Size([2, 24])
print(torch.equal(greedy[:, :8], prompt))        # True -> the prompt is echoed back
```

`generate` is `@torch.no_grad()`, drives the KV cache itself, and restores the
previous `training` flag when it is done.

### Streaming / chunked inference

Feed the sequence in pieces, carrying `states` **and** `caches`:

```python
import torch
from dialectic import DialecticTransformer

torch.manual_seed(0)
model = DialecticTransformer(vocab_size=100, d_model=64, num_layers=2,
                             num_heads=4, state_dim=32, chunk_size=8, mem_slots=4)
model.eval()
ids = torch.randint(0, 100, (2, 48))

with torch.no_grad():
    reference, _, _, _, _ = model(ids, episode_reset=True)

    pieces, states, caches = [], None, None
    for start in range(0, ids.shape[1], 5):          # deliberately ragged, 5 != chunk_size
        block = ids[:, start:start + 5]
        out, states, caches, _, _ = model(
            block, states=states, caches=caches,
            episode_reset=(start == 0), use_cache=True,
        )
        pieces.append(out)

streamed = torch.cat(pieces, dim=1)
print(torch.allclose(reference, streamed, rtol=1e-4, atol=1e-4))   # True
```

### Padding

```python
import torch
from dialectic import DialecticTransformer

torch.manual_seed(0)
model = DialecticTransformer(vocab_size=100, d_model=64, num_layers=2,
                             num_heads=4, state_dim=32, chunk_size=8, mem_slots=4)
model.eval()

ids = torch.randint(0, 100, (2, 16))
pad = torch.zeros(2, 16, dtype=torch.bool)
pad[1, 12:] = True                    # row 1 is 4 tokens shorter

with torch.no_grad():
    logits, states, _, _, _ = model(ids, episode_reset=True, key_padding_mask=pad)

print(logits.shape)                                   # torch.Size([2, 16, 100])
print(bool(torch.isfinite(logits).all()))             # True
```

Padded tokens are masked out of attention, excluded from the SSM accumulation,
and blocked from writing episodic slots — per row, so one short row in a batch
never contaminates its neighbours.

---

## Chunk-invariance: full model needs `use_cache=True`

This is the subtlety worth stating loudly, because it looks like a bug the
first time you hit it.

| What you split | `use_cache=False` | `use_cache=True` |
|---|---|---|
| `ChunkwiseSSM` branch alone | **invariant** (max abs diff ~1e-7) | invariant |
| Full `DialecticTransformer` | **not invariant** (max abs diff 7 to 22) | invariant (max abs diff ~1e-5) |

(Measured by splitting mid-chunk: ~7 on the 2-layer model used in the snippets
below, ~22 on the 4-layer demo config in `dialectic.py` — the error grows with
the amount of attention history that gets dropped.)

The recurrent branch needs no cache: its state *is* the history, so cutting the
sequence anywhere and carrying `RecurrentState` reproduces the full pass
exactly.

```python
import torch
from dialectic import DialecticTransformer

torch.manual_seed(0)
model = DialecticTransformer(vocab_size=100, d_model=64, num_layers=2,
                             num_heads=4, state_dim=32, chunk_size=8, mem_slots=4)
model.eval()
ids = torch.randint(0, 100, (2, 32))

with torch.no_grad():
    ssm = model.layers[0].ssm
    x = model.layers[0].norm1(model.embed(ids))       # the SSM branch on its own
    full, _ = ssm(x, None, True, None)
    pre, state = ssm(x[:, :5], None, True, None)      # split mid-chunk, no cache
    post, _ = ssm(x[:, 5:], state, False, None)

print(torch.allclose(full[:, :5], pre, atol=1e-5))    # True
print(torch.allclose(full[:, 5:], post, atol=1e-5))   # True -> no cache needed
```

Causal attention is the branch that does need the cache. Without it, a split
segment can only attend to itself, so the second half of the sequence loses all
of its history:

```python
import torch
from dialectic import DialecticTransformer

torch.manual_seed(0)
model = DialecticTransformer(vocab_size=100, d_model=64, num_layers=2,
                             num_heads=4, state_dim=32, chunk_size=8, mem_slots=4)
model.eval()
ids = torch.randint(0, 100, (2, 32))

with torch.no_grad():
    full, _, _, _, _ = model(ids, episode_reset=True)

    # WRONG: state carried, attention history dropped
    _, st, _, _, _ = model(ids[:, :5], episode_reset=True)
    bad, _, _, _, _ = model(ids[:, 5:], states=st, episode_reset=False)

    # RIGHT: carry the KV cache too
    _, st2, ca, _, _ = model(ids[:, :5], episode_reset=True, use_cache=True)
    good, _, _, _, _ = model(ids[:, 5:], states=st2, caches=ca,
                             episode_reset=False, use_cache=True)

print(torch.allclose(full[:, 5:], bad, atol=1e-4))    # False
print(torch.allclose(full[:, 5:], good, atol=1e-4))   # True
```

So: **carry `caches` alongside `states` whenever you split a sequence.**
`generate` already does this for you.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q          # 62 tests
python dialectic.py    # the end-to-end demo, prints "Hasil akhir: SEMUA OK"
```

Everything runs on CPU with fixed seeds, so the suite is deterministic in CI.

| File | Tests | Covers |
|---|---:|---|
| `tests/conftest.py` | — | shared seeded model/ids/targets fixtures and comparison helpers |
| `tests/test_consistency.py` | 10 | chunk invariance (SSM alone and full model with cache), empty sequence, `ignore_index`, memory carry, FIFO eviction, pad-id independence |
| `tests/test_episodic_memory.py` | 5 | read/write ablations proving memory changes the output, empty-buffer readout, detached writes, `Mem/*` metrics |
| `tests/test_generation.py` | 7 | greedy vs argmax, `top_k=1` ≡ greedy, seeded sampling, vocab bounds, `top_k` clamping, train-mode restore, no grad graph |
| `tests/test_gradients.py` | 9 | finite loss and non-zero grads on every trainable path, no KV cache while training, second backward after a state carry |
| `tests/test_masking_and_state.py` | 8 | 2-D bool/float masks, RoPE offset behaviour, `pos` bookkeeping, `episode_reset`, non-causal mode, dropout train/eval |
| `tests/test_mixed_phase.py` | 6 | `ChunkwiseSSM._run_mixed`: dispatch on ragged phases, and the contract that a mixed-phase batch equals the rows run separately |
| `tests/test_padding_paths.py` | 8 | 3-D and 4-D attention masks, fully padded chunks, and per-row episodic write masks |
| `tests/test_validation.py` | 9 | constructor and argument validation raising early and loudly |

Two of those files are worth a note on *why* they are shaped the way they are.

**`test_mixed_phase.py`** targets `_run_mixed`, the slow path used when rows of a
batch sit at different chunk phases — what continuous batching produces when
prompts have different lengths. The contract is simple and strict: *a mixed
batch must be identical to processing each row on its own*, in the outputs and
in every field of the returned state. A batch whose rows share a phase never
leaves `_run_aligned` and so can never catch a bug in that path.

**`test_padding_paths.py`** covers the mask plumbing, and its write-mask tests
deliberately use a **mixed batch** (one real row, one fully padded row). This
matters: when *every* row of a chunk is padding, the SSM skips it through a
frozen fast path and no write happens regardless of the mask — so a uniform
batch would still pass even if the per-row write mask had been collapsed into a
single batch-wide flag. Only a mixed batch distinguishes "write for the rows
that carry real tokens" from "write for everyone".

---

## Files

| Path | What it is |
|---|---|
| `dialectic.py` | the whole model plus `run_demo()`, which CI runs end to end |
| `tests/` | the pytest suite described above |
| `HASIL_TEST.md` | the original test report (in Indonesian), including the chunk-invariance investigation |
| `.github/workflows/ci.yml` | pytest on Python 3.10 / 3.11 / 3.12 plus the demo script |
