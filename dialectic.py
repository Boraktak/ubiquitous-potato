import math
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentState(NamedTuple):
    hidden: torch.Tensor
    memory: torch.Tensor
    mem_valid: torch.Tensor
    pos: torch.Tensor
    phase: torch.Tensor
    chunk_drift: torch.Tensor

    @staticmethod
    def zeros(
        batch: int,
        state_dim: int,
        mem_slots: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "RecurrentState":
        return RecurrentState(
            hidden=torch.zeros(batch, state_dim, device=device, dtype=dtype),
            memory=torch.zeros(batch, mem_slots, state_dim, device=device, dtype=dtype),
            mem_valid=torch.zeros(batch, mem_slots, device=device, dtype=torch.bool),
            pos=torch.zeros(batch, device=device, dtype=torch.float32),
            phase=torch.zeros(batch, device=device, dtype=torch.long),
            chunk_drift=torch.zeros(batch, state_dim, device=device, dtype=dtype),
        )

    def detach_to(self, device: torch.device, dtype: torch.dtype) -> "RecurrentState":
        return RecurrentState(
            hidden=self.hidden.detach().to(device=device, dtype=dtype),
            memory=self.memory.detach().to(device=device, dtype=dtype),
            mem_valid=self.mem_valid.detach().to(device=device),
            pos=self.pos.detach().to(device=device, dtype=torch.float32),
            phase=self.phase.detach().to(device=device, dtype=torch.long),
            chunk_drift=self.chunk_drift.detach().to(device=device, dtype=dtype),
        )


class LayerCache(NamedTuple):
    k: torch.Tensor
    v: torch.Tensor
    key_pad: Optional[torch.Tensor]


class TemporalRoPE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        if pos.dim() == 1:
            pos = pos.unsqueeze(0).expand(x.shape[0], -1)
        freqs = pos.unsqueeze(-1).float() * self.inv_freq.float()
        cos = freqs.cos().unsqueeze(1).repeat_interleave(2, dim=-1)
        sin = freqs.sin().unsqueeze(1).repeat_interleave(2, dim=-1)
        xf = x.float()
        x1, x2 = xf[..., ::2], xf[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return (xf * cos + x_rot * sin).to(dtype=x.dtype)


class EpisodicMemory(nn.Module):
    """Content-addressed FIFO slots.

    Write is detached on purpose (truncated BPTT / no credit through stored slots).
    Retrieve output is memory-only: state is the query, never concatenated back.
    """

    def __init__(self, state_dim: int, num_slots: int = 16):
        super().__init__()
        if num_slots < 1:
            raise ValueError("num_slots must be >= 1")
        self.num_slots = num_slots
        self.query_proj = nn.Linear(state_dim, state_dim, bias=False)
        self.key_proj = nn.Linear(state_dim, state_dim, bias=False)
        self.val_proj = nn.Linear(state_dim, state_dim, bias=False)
        self.out_proj = nn.Linear(state_dim, state_dim, bias=False)
        self.inject = nn.Linear(state_dim, state_dim, bias=False)
        with torch.no_grad():
            self.inject.weight.copy_(torch.eye(state_dim) * 1e-3)

    def retrieve(self, state: torch.Tensor, buffer: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(state).unsqueeze(1)
        k = self.key_proj(buffer)
        v = self.val_proj(buffer)
        scale = state.shape[-1] ** -0.5
        scores = torch.bmm(q, k.transpose(1, 2)) * scale
        scores = scores.masked_fill(~valid.unsqueeze(1), torch.finfo(scores.dtype).min)
        empty = ~valid.any(dim=-1)
        scores = scores.masked_fill(empty.view(-1, 1, 1), 0.0)
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(empty.view(-1, 1, 1), 0.0)
        mem = torch.bmm(attn, v).squeeze(1)
        return self.out_proj(mem)

    def write(
        self, state: torch.Tensor, buffer: torch.Tensor, valid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        new_buf = torch.cat([buffer[:, 1:], state.detach().unsqueeze(1)], dim=1)
        ones = torch.ones(valid.shape[0], 1, device=valid.device, dtype=torch.bool)
        new_valid = torch.cat([valid[:, 1:], ones], dim=1)
        return new_buf, new_valid


def _as_additive(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if mask.dtype == torch.bool or not mask.is_floating_point():
        keep = mask.to(dtype=torch.bool)
        return torch.zeros(keep.shape, device=mask.device, dtype=dtype).masked_fill(~keep, float("-inf"))
    return mask.to(dtype=dtype)


def _sdpa_mask(
    attn_mask: Optional[torch.Tensor],
    key_padding_mask: Optional[torch.Tensor],
    q_len: int,
    kv_len: int,
    cache_len: int,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    causal: bool,
) -> Tuple[Optional[torch.Tensor], bool]:
    if attn_mask is not None and key_padding_mask is None and attn_mask.dim() == 2:
        if attn_mask.shape in {(batch, q_len), (batch, kv_len)}:
            key_padding_mask, attn_mask = attn_mask, None
    if causal and cache_len == 0 and attn_mask is None and key_padding_mask is None:
        return None, True
    if (not causal) and attn_mask is None and key_padding_mask is None:
        return None, False
    if causal and cache_len > 0 and q_len == 1 and attn_mask is None and key_padding_mask is None:
        return None, False

    add: Optional[torch.Tensor] = None
    if causal and not (q_len <= 1 and cache_len == 0):
        q_idx = torch.arange(q_len, device=device) + cache_len
        k_idx = torch.arange(kv_len, device=device)
        add = torch.zeros(q_len, kv_len, device=device, dtype=dtype)
        add = add.masked_fill(q_idx[:, None] < k_idx[None, :], float("-inf"))
    if attn_mask is not None:
        m = attn_mask
        if m.dim() == 2 and m.shape[-2] == q_len and m.shape[-1] == kv_len:
            am = _as_additive(m, dtype)
        elif m.dim() == 2 and m.shape[-1] == kv_len:
            am = _as_additive(m, dtype)[:, None, None, :]
        elif m.dim() == 3:
            am = _as_additive(m, dtype).unsqueeze(1)
        else:
            am = _as_additive(m, dtype)
        add = am if add is None else add + am
    if key_padding_mask is not None:
        pad = key_padding_mask.to(device=device, dtype=torch.bool)
        if pad.shape[-1] != kv_len and pad.shape[-1] == q_len and cache_len > 0:
            pad = torch.cat(
                [torch.zeros(pad.shape[0], cache_len, dtype=torch.bool, device=device), pad],
                dim=-1,
            )
        if pad.shape[-1] != kv_len:
            raise ValueError(f"key_padding_mask last dim {pad.shape[-1]} != kv_len {kv_len}")
        km = torch.zeros(pad.shape[0], 1, 1, pad.shape[-1], device=device, dtype=dtype)
        km = km.masked_fill(pad[:, None, None, :], float("-inf"))
        add = km if add is None else add + km
    return add, False


class ChunkwiseSSM(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        state_dim: int = 256,
        chunk_size: int = 32,
        mem_slots: int = 16,
    ):
        super().__init__()
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.mem_slots = mem_slots
        self.input_proj = nn.Linear(d_model, state_dim)
        self.f_drift = nn.Linear(state_dim, state_dim)
        self.forget_gate = nn.Linear(state_dim, state_dim)
        self.readout = nn.Linear(state_dim, d_model)
        self.state_norm = nn.LayerNorm(state_dim)
        self.episodic = EpisodicMemory(state_dim, mem_slots)
        self._input_inject = nn.Parameter(torch.tensor(math.log(math.expm1(0.1))))
        nn.init.xavier_uniform_(self.f_drift.weight, gain=0.1)
        nn.init.zeros_(self.f_drift.bias)
        nn.init.xavier_uniform_(self.forget_gate.weight, gain=0.1)
        self.last_stats: Dict[str, float] = {}

    def _init_or_cast(
        self,
        prev: Optional[RecurrentState],
        episode_reset: bool,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RecurrentState:
        bad = (
            prev is None
            or episode_reset
            or prev.hidden.shape[0] != batch
            or prev.hidden.shape[-1] != self.state_dim
            or prev.memory.shape[1] != self.mem_slots
        )
        if bad:
            return RecurrentState.zeros(batch, self.state_dim, self.mem_slots, device, dtype)
        return prev.detach_to(device, dtype)

    def _gate(self, raw_seq: torch.Tensor) -> torch.Tensor:
        normed = self.state_norm(raw_seq)
        forget = torch.sigmoid(self.forget_gate(normed))
        return torch.tanh(normed * forget)

    def _accumulate(
        self,
        u: torch.Tensor,
        raw0: torch.Tensor,
        drift: torch.Tensor,
        valid: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inject = F.softplus(self._input_inject).to(dtype=u.dtype)
        inc = drift.unsqueeze(1) + inject * u
        if valid is not None:
            inc = inc * valid.unsqueeze(-1).to(dtype=inc.dtype)
        raw_seq = raw0.unsqueeze(1) + torch.cumsum(inc, dim=1)
        gated = self._gate(raw_seq)
        return self.readout(gated), raw_seq[:, -1, :], gated[:, -1, :]

    def _open_chunk(
        self, state: torch.Tensor, memory: torch.Tensor, mem_valid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        retrieved = self.episodic.retrieve(state, memory, mem_valid)
        injected = self.episodic.inject(retrieved)
        state = state + injected
        step = 1.0 / math.sqrt(self.chunk_size)
        drift = self.f_drift(self.state_norm(state)) * step
        self._stat_n += 1
        self._stat_read += retrieved.detach().norm(dim=-1).mean().item()
        self._stat_inj += injected.detach().norm(dim=-1).mean().item()
        self._stat_valid += mem_valid.float().sum(dim=-1).mean().item()
        return state, drift

    def _finish_chunk(
        self,
        gated_last: torch.Tensor,
        memory: torch.Tensor,
        mem_valid: torch.Tensor,
        do_write: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_mem, new_valid = self.episodic.write(gated_last, memory, mem_valid)
        keep = do_write.view(-1, 1, 1)
        memory = torch.where(keep, new_mem, memory)
        mem_valid = torch.where(do_write.view(-1, 1), new_valid, mem_valid)
        return gated_last, memory, mem_valid

    def _segment_write_mask(self, seg_v: Optional[torch.Tensor], batch: int, device: torch.device) -> torch.Tensor:
        if seg_v is None:
            return torch.ones(batch, dtype=torch.bool, device=device)
        return seg_v.any(dim=-1)

    def _run_aligned(
        self,
        u: torch.Tensor,
        state: torch.Tensor,
        memory: torch.Tensor,
        mem_valid: torch.Tensor,
        phase: int,
        drift: torch.Tensor,
        valid: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        b, s, _ = u.shape
        outs: List[torch.Tensor] = []
        t = 0

        if phase > 0 and s > 0:
            take = min(self.chunk_size - phase, s)
            seg_v = None if valid is None else valid[:, t : t + take]
            out, raw_last, gated_last = self._accumulate(u[:, t : t + take], state, drift, seg_v)
            outs.append(out)
            t += take
            phase += take
            if phase == self.chunk_size:
                do_write = self._segment_write_mask(seg_v, b, u.device)
                state, memory, mem_valid = self._finish_chunk(gated_last, memory, mem_valid, do_write)
                phase = 0
                drift = torch.zeros_like(state)
            else:
                state = raw_last

        while t < s:
            take = min(self.chunk_size, s - t)
            seg_v = None if valid is None else valid[:, t : t + take]
            all_pad = seg_v is not None and (not bool(seg_v.any().item()))
            if all_pad:
                frozen = self._gate(state.unsqueeze(1).expand(-1, take, -1))
                outs.append(self.readout(frozen))
                t += take
                continue

            state, drift = self._open_chunk(state, memory, mem_valid)
            out, raw_last, gated_last = self._accumulate(u[:, t : t + take], state, drift, seg_v)
            outs.append(out)
            t += take
            if take == self.chunk_size:
                do_write = self._segment_write_mask(seg_v, b, u.device)
                state, memory, mem_valid = self._finish_chunk(gated_last, memory, mem_valid, do_write)
                phase = 0
                drift = torch.zeros_like(state)
            else:
                state = raw_last
                phase = take

        if not outs:
            empty = u.new_zeros(b, 0, self.readout.out_features)
            return empty, state, memory, mem_valid, phase, drift
        return torch.cat(outs, dim=1), state, memory, mem_valid, phase, drift

    def _run_mixed(
        self,
        u: torch.Tensor,
        state: torch.Tensor,
        memory: torch.Tensor,
        mem_valid: torch.Tensor,
        phase: torch.Tensor,
        drift: torch.Tensor,
        valid: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Slow path: per-sample phases differ (e.g. continuous batching).
        b, s, _ = u.shape
        outs = []
        new_state = state.clone()
        new_memory = memory.clone()
        new_valid = mem_valid.clone()
        new_phase = phase.clone()
        new_drift = drift.clone()
        for i in range(b):
            v_i = None if valid is None else valid[i : i + 1]
            out, st, mem, mv, ph, dr = self._run_aligned(
                u[i : i + 1],
                state[i : i + 1],
                memory[i : i + 1],
                mem_valid[i : i + 1],
                int(phase[i].item()),
                drift[i : i + 1],
                v_i,
            )
            outs.append(out)
            new_state[i : i + 1] = st
            new_memory[i : i + 1] = mem
            new_valid[i : i + 1] = mv
            new_phase[i] = ph
            new_drift[i : i + 1] = dr
        if s == 0:
            empty = u.new_zeros(b, 0, self.readout.out_features)
            return empty, new_state, new_memory, new_valid, new_phase, new_drift
        return torch.cat(outs, dim=0), new_state, new_memory, new_valid, new_phase, new_drift

    def forward(
        self,
        x_norm: torch.Tensor,
        prev: Optional[RecurrentState] = None,
        episode_reset: bool = False,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, RecurrentState]:
        b, s, _ = x_norm.shape
        rec = self._init_or_cast(prev, episode_reset, b, x_norm.device, x_norm.dtype)
        self._stat_n = 0
        self._stat_read = 0.0
        self._stat_inj = 0.0
        self._stat_valid = 0.0
        if s == 0:
            self.last_stats = {"opens": 0.0, "read_norm": 0.0, "inject_norm": 0.0, "valid_slots": 0.0}
            return x_norm.new_zeros(b, 0, self.readout.out_features), rec

        u = self.input_proj(x_norm)
        valid = None if valid_mask is None else valid_mask.to(device=x_norm.device, dtype=torch.bool)
        if bool(torch.all(rec.phase == rec.phase[:1])):
            out, state, memory, mem_valid, ph, drift = self._run_aligned(
                u, rec.hidden, rec.memory, rec.mem_valid, int(rec.phase[0].item()), rec.chunk_drift, valid
            )
            phase_out = rec.phase.new_full((b,), ph)
        else:
            out, state, memory, mem_valid, phase_out, drift = self._run_mixed(
                u, rec.hidden, rec.memory, rec.mem_valid, rec.phase, rec.chunk_drift, valid
            )
        n = max(self._stat_n, 1)
        self.last_stats = {
            "opens": float(self._stat_n),
            "read_norm": self._stat_read / n,
            "inject_norm": self._stat_inj / n,
            "valid_slots": self._stat_valid / n,
        }
        return out, RecurrentState(state, memory, mem_valid, rec.pos, phase_out, drift)


class HybridDialecticLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        state_dim: int = 256,
        dropout: float = 0.1,
        chunk_size: int = 32,
        mem_slots: int = 16,
        causal: bool = True,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.causal = causal
        self.rope = TemporalRoPE(self.d_head)
        self.ssm = ChunkwiseSSM(d_model, state_dim, chunk_size, mem_slots)
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.fusion_gate = nn.Linear(d_model * 2, d_model)
        nn.init.constant_(self.fusion_gate.bias, -1.0)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def _start_pos(
        self,
        prev: Optional[RecurrentState],
        episode_reset: bool,
        batch: int,
        device: torch.device,
        pos_offset: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        if prev is None or episode_reset or prev.pos.shape[0] != batch:
            start = torch.zeros(batch, device=device, dtype=torch.float32)
        else:
            start = prev.pos.detach().to(device=device, dtype=torch.float32)
        if torch.is_tensor(pos_offset):
            start = start + pos_offset.to(device=device, dtype=torch.float32)
        else:
            start = start + float(pos_offset)
        return start

    def forward(
        self,
        x: torch.Tensor,
        prev_state: Optional[RecurrentState] = None,
        prev_cache: Optional[LayerCache] = None,
        episode_reset: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        pos_offset: Union[float, torch.Tensor] = 0.0,
        use_cache: bool = False,
    ):
        b, s, d = x.shape
        x_norm = self.norm1(x)
        valid = None
        if key_padding_mask is not None:
            valid = ~key_padding_mask.to(device=x.device, dtype=torch.bool)
        ssm_out, new_state = self.ssm(x_norm, prev_state, episode_reset, valid_mask=valid)
        if s == 0:
            return x, new_state, None

        start_pos = self._start_pos(prev_state, episode_reset, b, x.device, pos_offset)
        positions = start_pos.unsqueeze(1) + torch.arange(s, device=x.device, dtype=torch.float32)
        qkv = self.qkv(x_norm).view(b, s, 3, self.num_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = self.rope(q, positions), self.rope(k, positions)

        cache_len = 0
        cur_pad = None if key_padding_mask is None else key_padding_mask.to(device=x.device, dtype=torch.bool)
        full_pad = cur_pad
        if use_cache and prev_cache is not None and not episode_reset:
            if (
                prev_cache.k.shape[0] == b
                and prev_cache.k.shape[1] == self.num_heads
                and prev_cache.k.shape[-1] == self.d_head
            ):
                cache_len = prev_cache.k.shape[2]
                k = torch.cat([prev_cache.k, k], dim=2)
                v = torch.cat([prev_cache.v, v], dim=2)
                prev_pad = prev_cache.key_pad
                if prev_pad is None:
                    prev_pad = torch.zeros(b, cache_len, dtype=torch.bool, device=x.device)
                now_pad = cur_pad if cur_pad is not None else torch.zeros(b, s, dtype=torch.bool, device=x.device)
                full_pad = torch.cat([prev_pad, now_pad], dim=-1)

        drop_p = self.dropout.p if self.training else 0.0
        mask, is_causal = _sdpa_mask(
            attn_mask, full_pad, s, k.shape[2], cache_len, b, x.device, q.dtype, self.causal
        )
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop_p, is_causal=is_causal)
        attn = self.o_proj(attn.transpose(1, 2).contiguous().view(b, s, d))
        gate = torch.sigmoid(self.fusion_gate(torch.cat([ssm_out, attn], dim=-1)))
        x = x + self.dropout(gate * ssm_out + (1.0 - gate) * attn)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        new_state = new_state._replace(pos=(start_pos + s).detach())
        new_cache = None
        if use_cache:
            kp = None if full_pad is None else full_pad.detach()
            new_cache = LayerCache(k.detach(), v.detach(), kp)
        return x, new_state, new_cache


class DialecticTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        state_dim: int = 128,
        dropout: float = 0.1,
        chunk_size: int = 32,
        mem_slots: int = 16,
        causal: bool = True,
        ignore_index: int = -100,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.chunk_size = chunk_size
        self.mem_slots = mem_slots
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                HybridDialecticLayer(d_model, num_heads, state_dim, dropout, chunk_size, mem_slots, causal)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.embed.weight

    def forward(
        self,
        ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        states: Optional[List[RecurrentState]] = None,
        caches: Optional[List[Optional[LayerCache]]] = None,
        episode_reset: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        pos_offset: Union[float, torch.Tensor] = 0.0,
        use_cache: bool = False,
    ) -> Tuple[
        torch.Tensor,
        List[RecurrentState],
        Optional[List[LayerCache]],
        Optional[torch.Tensor],
        Dict[str, float],
    ]:
        n = len(self.layers)
        if states is not None and len(states) != n:
            raise ValueError(f"states has {len(states)} entries, expected {n}")
        if caches is not None and len(caches) != n:
            raise ValueError(f"caches has {len(caches)} entries, expected {n}")

        x = self.embed(ids)
        new_states: List[RecurrentState] = []
        new_caches: List[LayerCache] = []
        for i, layer in enumerate(self.layers):
            x, ns, nc = layer(
                x,
                prev_state=None if states is None else states[i],
                prev_cache=None if caches is None else caches[i],
                episode_reset=episode_reset,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                pos_offset=pos_offset,
                use_cache=use_cache,
            )
            new_states.append(ns)
            if nc is not None:
                new_caches.append(nc)

        logits = self.lm_head(self.norm(x))
        metrics: Dict[str, float] = {}
        for i, layer in enumerate(self.layers):
            for key, val in layer.ssm.last_stats.items():
                metrics[f"Mem/L{i}_{key}"] = val
        total_loss = None
        if targets is not None:
            v = logits.shape[-1]
            total_loss = F.cross_entropy(
                logits.reshape(-1, v),
                targets.reshape(-1),
                ignore_index=self.ignore_index,
            )
            metrics["Loss/CE"] = total_loss.item()
            metrics["Loss/Total"] = total_loss.item()
        return logits, new_states, (new_caches if use_cache else None), total_loss, metrics

    @torch.no_grad()
    def generate(
        self,
        ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        episode_reset: bool = True,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        logits, states, caches, _, _ = self.forward(ids, episode_reset=episode_reset, use_cache=True)
        out = [ids]
        for _ in range(max_new_tokens):
            step = logits[:, -1, :]
            if temperature <= 0.0:
                nxt = step.argmax(dim=-1, keepdim=True)
            else:
                step = step / temperature
                if top_k is not None and top_k > 0:
                    kth = min(top_k, step.size(-1))
                    thresh = torch.topk(step, kth, dim=-1).values[:, -1:]
                    step = step.masked_fill(step < thresh, float("-inf"))
                nxt = torch.multinomial(F.softmax(step, dim=-1), 1)
            out.append(nxt)
            logits, states, caches, _, _ = self.forward(
                nxt, states=states, caches=caches, episode_reset=False, use_cache=True
            )
        if was_training:
            self.train()
        return torch.cat(out, dim=1)


def _grad_ok(p: torch.Tensor) -> bool:
    g = p.grad
    return g is not None and bool(torch.isfinite(g).all().item()) and g.abs().sum().item() > 0.0


def _close(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-4) -> bool:
    return torch.allclose(a, b, rtol=rtol, atol=atol)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def _status(ok: bool) -> str:
    return "OK" if ok else "Gagal"


def run_demo() -> bool:
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device  : {device}")
    model = DialecticTransformer(
        vocab_size=1000,
        d_model=256,
        num_layers=4,
        num_heads=4,
        state_dim=128,
        chunk_size=32,
        mem_slots=16,
    ).to(device)

    ids = torch.randint(0, 1000, (2, 256), device=device)
    targets = torch.randint(0, 1000, (2, 256), device=device)
    print("Menjalankan A1 + episodic (chunk-invariant, CE only)...")
    model.train()
    _, states, caches, total_loss, metrics = model(ids, targets=targets, episode_reset=True)

    print("\nLoss & Mem stats:")
    for key in ("Loss/CE", "Loss/Total"):
        if key in metrics:
            print(f"   - {key:<22} : {metrics[key]:.4f}")
    for key, val in metrics.items():
        if key.startswith("Mem/"):
            print(f"   - {key:<22} : {val:.6f}")

    model.zero_grad(set_to_none=True)
    assert total_loss is not None
    total_loss.backward()
    ok_ssm = all(_grad_ok(layer.ssm.f_drift.weight) for layer in model.layers)
    ok_mem = all(_grad_ok(layer.ssm.episodic.out_proj.weight) for layer in model.layers)
    ok_inj = all(_grad_ok(layer.ssm.episodic.inject.weight) for layer in model.layers)
    ok_q = all(_grad_ok(layer.ssm.episodic.query_proj.weight) for layer in model.layers)
    ok_k = all(_grad_ok(layer.ssm.episodic.key_proj.weight) for layer in model.layers)
    ok_v = all(_grad_ok(layer.ssm.episodic.val_proj.weight) for layer in model.layers)

    second_ok = False
    try:
        _, _, _, loss2, _ = model(ids, targets=targets, states=states, episode_reset=False)
        model.zero_grad(set_to_none=True)
        assert loss2 is not None
        loss2.backward()
        second_ok = True
    except RuntimeError:
        second_ok = False

    model.eval()
    with torch.no_grad():
        empty_ok = model(ids[:, :0], episode_reset=True)[0].shape == (ids.shape[0], 0, 1000)
        masked = targets.clone()
        masked[:, :8] = -100
        mask_ok = bool(torch.isfinite(model(ids, targets=masked, episode_reset=True)[3]))
        train_cache_ok = caches is None

        _, states2, _, _, _ = model(ids, states=states, episode_reset=False)
        mem_carry_ok = all(bool(s.mem_valid.any().item()) for s in states) and all(
            not torch.equal(a.memory, b.memory) for a, b in zip(states, states2)
        )
        gen_ok = model.generate(ids[:, :8], max_new_tokens=4, temperature=0.0).shape == (ids.shape[0], 12)

        # Full-model chunk invariance requires the KV cache: the causal attention
        # branch must see all previous keys, otherwise splitting drops history.
        full, _, _, _, _ = model(ids, episode_reset=True)
        pre, st, ca_mid, _, _ = model(ids[:, :8], episode_reset=True, use_cache=True)
        post, _, _, _, _ = model(ids[:, 8:], states=st, caches=ca_mid, episode_reset=False, use_cache=True)
        split_mid = _close(full[:, :8], pre) and _close(full[:, 8:], post)

        pre32, st32, ca_b, _, _ = model(ids[:, :32], episode_reset=True, use_cache=True)
        post32, _, _, _, _ = model(ids[:, 32:], states=st32, caches=ca_b, episode_reset=False, use_cache=True)
        split_bound = _close(full[:, :32], pre32) and _close(full[:, 32:], post32)

        # The SSM branch alone is chunk-invariant even WITHOUT cache
        # (this is the true "chunk-invariant path").
        ssm0 = model.layers[0].ssm
        xin = model.embed(ids)
        xn = model.layers[0].norm1(xin)
        s_full, _ = ssm0(xn, None, True, None)
        s_pre, s_st = ssm0(xn[:, :8], None, True, None)
        s_post, _ = ssm0(xn[:, 8:], s_st, False, None)
        ssm_split = _close(s_full[:, :8], s_pre) and _close(s_full[:, 8:], s_post)

        full_c, _, _, _, _ = model(ids, episode_reset=True, use_cache=True)
        pre_c, st_c, ca, _, _ = model(ids[:, :8], episode_reset=True, use_cache=True)
        post_c, _, _, _, _ = model(ids[:, 8:], states=st_c, caches=ca, episode_reset=False, use_cache=True)
        cache_split = _close(full_c[:, :8], pre_c) and _close(full_c[:, 8:], post_c)

        ids2 = torch.randint(0, 1000, (2, 64), device=device)
        live, _, _, _, _ = model(ids2, states=states, episode_reset=False)
        zero_st = [
            s._replace(memory=torch.zeros_like(s.memory), mem_valid=torch.zeros_like(s.mem_valid))
            for s in states
        ]
        dead, _, _, _, _ = model(ids2, states=zero_st, episode_reset=False)
        shuf_st = [
            s._replace(memory=s.memory.roll(1, dims=0), mem_valid=s.mem_valid.roll(1, dims=0))
            for s in states
        ]
        shuf, _, _, _, _ = model(ids2, states=shuf_st, episode_reset=False)

        layer0 = model.layers[0].ssm.episodic
        s0 = states[0]
        read_live = layer0.retrieve(s0.hidden, s0.memory, s0.mem_valid)
        read_dead = layer0.retrieve(s0.hidden, torch.zeros_like(s0.memory), torch.zeros_like(s0.mem_valid))
        retrieve_ok = read_live.norm() > 0 and read_dead.abs().max().item() == 0.0
        zero_diff = _max_abs(live, dead)
        shuf_diff = _max_abs(live, shuf)
        mem_affects = retrieve_ok and zero_diff > 1e-6
        mem_shuffle = shuf_diff > 1e-6

        pad = torch.zeros(ids.shape, dtype=torch.bool, device=device)
        pad[:, -4:] = True
        ids_a = ids.clone()
        ids_b = ids.clone()
        ids_a[:, -4:] = 0
        ids_b[:, -4:] = 1
        logits_a, st_a, _, _, _ = model(ids_a, episode_reset=True, key_padding_mask=pad)
        logits_b, st_b, _, _, _ = model(ids_b, episode_reset=True, key_padding_mask=pad)
        pad_indep = _close(logits_a[:, :-4], logits_b[:, :-4]) and _close(st_a[0].hidden, st_b[0].hidden)
        pad_finite = bool(torch.isfinite(logits_a).all().item())

        long = torch.randint(0, 1000, (2, model.chunk_size * (model.mem_slots + 4)), device=device)
        _, long_states, _, _, _ = model(long, episode_reset=True)
        evict_ok = all(bool(st.mem_valid.all().item()) for st in long_states)

    print("\nGradien:")
    print(f"   - SSM drift           : {_status(ok_ssm)}")
    print(f"   - Memory out          : {_status(ok_mem)}")
    print(f"   - Mem inject          : {_status(ok_inj)}")
    print(f"   - Query/Key/Value     : {_status(ok_q and ok_k and ok_v)}")
    print(f"   - Carry + 2nd backward: {_status(second_ok)}")
    print("\nKonsistensi:")
    print(f"   - State carry (mem)   : {_status(mem_carry_ok)}")
    print(f"   - No KV in training   : {_status(train_cache_ok)}")
    print(f"   - S=0                 : {_status(empty_ok)}")
    print(f"   - ignore_index        : {_status(mask_ok)}")
    print(f"   - generate            : {_status(gen_ok)}")
    print(f"   - Split mid-chunk     : {_status(split_mid)}")
    print(f"   - Split chunk-bound   : {_status(split_bound)}")
    print(f"   - SSM-only split (nocache): {_status(ssm_split)}")
    print(f"   - Cache split         : {_status(cache_split)}")
    print(f"   - Pad-id independence : {_status(pad_indep and pad_finite)}")
    print(f"   - Eviction FIFO       : {_status(evict_ok)}")
    print("\nAblasi episodic:")
    print(f"   - Retrieve live/zero  : {_status(retrieve_ok)}")
    print(f"   - Zero-mem max|d|     : {zero_diff:.3e} -> {_status(mem_affects)}")
    print(f"   - Shuffle-mem max|d|  : {shuf_diff:.3e} -> {_status(mem_shuffle)}")

    checks = [
        ok_ssm, ok_mem, ok_inj, ok_q, ok_k, ok_v, second_ok,
        mem_carry_ok, train_cache_ok, empty_ok, mask_ok, gen_ok,
        split_mid, split_bound, ssm_split, cache_split, pad_indep, pad_finite,
        evict_ok, mem_affects, mem_shuffle,
    ]
    all_ok = all(checks)
    print("\nHasil akhir:", "SEMUA OK" if all_ok else "ADA YANG GAGAL")
    return all_ok


if __name__ == "__main__":
    ok = run_demo()
    raise SystemExit(0 if ok else 1)
