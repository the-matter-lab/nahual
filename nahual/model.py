# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nahual architecture, EDM losses and cached autoregressive sampling.

NextTokenDiffusion: full MoT trunk + causal AdaLN transformer.
DiffusionMLP: the same trunk + independent per-token AdaLN MLP.
State-dict names match the released paper checkpoints.

Adapted code (see THIRD_PARTY_NOTICES.md): the Qwen3 layers (Qwen3RMSNorm, Qwen3MLP,
rotate_half, apply_rotary_pos_emb, Qwen3RotaryEmbedding, Qwen3Attention) follow Hugging
Face transformers' models/qwen3/modeling_qwen3.py (Apache-2.0, Copyright 2025 The Qwen
team, Alibaba Group and the HuggingFace Inc. team). ResBlock and DiffMLPFinalLayer follow
MAR's models/diffloss.py (https://github.com/LTH14/mar; MIT, Copyright (c) 2024 Tianhong
Li). MPFourier is from NVlabs/edm2 (Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import warnings
import tqdm
from safetensors.torch import safe_open

from nahual.tokens import has_coords, coord_component_mask

from torch.nn.attention.flex_attention import BlockMask, flex_attention

try:  # torch >= 2.13. Absent on older releases.
    from torch.nn.attention.varlen import varlen_attn

    VARLEN_ATTENTION_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the installed torch
    varlen_attn = None
    VARLEN_ATTENTION_AVAILABLE = False
from nahual.tokens import GEN, STOP

ATTENTION_BACKENDS = ("flex", "varlen")
_warned_varlen_fallback = False


def resolve_attention_backend(backend):
    """Return the attention backend this torch build can run.

    "varlen" needs torch >= 2.13. Older torch falls back to "flex", which computes
    the same attention, and warns once per process.
    """
    global _warned_varlen_fallback
    if backend not in ATTENTION_BACKENDS:
        raise ValueError(f"attention_backend must be one of {ATTENTION_BACKENDS}, got {backend!r}")
    if backend == "varlen" and not VARLEN_ATTENTION_AVAILABLE:
        if not _warned_varlen_fallback:
            warnings.warn(
                f"attention_backend='varlen' needs torch>=2.13, but torch {torch.__version__} "
                f"is installed. Using 'flex', which computes the same attention. "
                f"Pass --attention_backend=flex to silence this warning.",
                RuntimeWarning,
                stacklevel=3,
            )
            _warned_varlen_fallback = True
        return "flex"
    return backend

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def dense_to_ordered(dense_blockmask):
    num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
    indices = (
        dense_blockmask.argsort(dim=-1, descending=False, stable=True)
        .flip(-1)
        .to(torch.int32)
    )
    return num_blocks[None, None].contiguous(), indices[None, None].contiguous()


def create_cu_seqlens(document_ids):
    """Document boundaries [0, start_1, ..., start_k, seq_len] (int32) for varlen attention.

    document_ids is the [seq_len] id vector of one packed chunk. torch.nonzero has a
    data-dependent shape, so call this outside fullgraph-compiled code.
    """
    seq_len = document_ids.shape[0]
    if seq_len == 0:
        raise ValueError("document_ids must be non-empty")
    starts = torch.nonzero(document_ids[1:] != document_ids[:-1], as_tuple=False).flatten() + 1
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=document_ids.device),
            starts.to(torch.int32),
            torch.full((1,), seq_len, dtype=torch.int32, device=document_ids.device),
        ]
    )


def varlen_attention(q, k, v, cu_seqlens, max_seqlen):
    """Causal varlen attention over one packed [1, heads, seq, head_dim] chunk.

    k and v may have fewer heads than q (GQA).
    """
    batch, num_heads, seq_len, head_dim = q.shape
    if batch != 1:
        raise ValueError(f"varlen attention expects one packed chunk, got batch={batch}")
    packed_q = q.transpose(1, 2).reshape(-1, num_heads, head_dim)
    packed_k = k.transpose(1, 2).reshape(-1, k.shape[1], head_dim)
    packed_v = v.transpose(1, 2).reshape(-1, v.shape[1], head_dim)
    out = varlen_attn(
        packed_q,
        packed_k,
        packed_v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        window_size=(-1, 0),  # causal
        enable_gqa=True,      # Qwen3 uses 16 query heads over 8 kv heads
    )
    return out.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)


def create_document_block_mask(document_ids):
    block_size = 128
    seq_len = document_ids.shape[0]
    if seq_len == 0:
        raise ValueError("document_ids must be non-empty")

    pad_len = (-seq_len) % block_size
    if pad_len > 0:
        document_ids_padded = torch.cat(
            [document_ids, document_ids[-1:].expand(pad_len)],
            dim=0,
        )
    else:
        document_ids_padded = document_ids

    num_blocks = document_ids_padded.numel() // block_size
    block_idx = torch.arange(num_blocks, dtype=torch.int32, device=document_ids.device)
    causal_blockmask_any = block_idx[:, None] >= block_idx
    causal_blockmask_all = block_idx[:, None] > block_idx

    docs_low = document_ids_padded.view(num_blocks, block_size)[:, 0].contiguous()
    docs_high = document_ids_padded.view(num_blocks, block_size)[:, -1].contiguous()
    document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
    document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)

    blockmask_any = causal_blockmask_any & document_blockmask_any
    blockmask_all = causal_blockmask_all & document_blockmask_all

    partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
    full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)

    def document_causal_mask(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = document_ids[q_idx] == document_ids[kv_idx]
        return causal_mask & document_mask

    return BlockMask.from_kv_blocks(
        partial_kv_num_blocks,
        partial_kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        BLOCK_SIZE=block_size,
        mask_mod=document_causal_mask,
        seq_lengths=(seq_len, seq_len),
    )

class FourierCoords(nn.Module):
    def __init__(self, in_features, num_channels, out_features, bandwidth=1):
        super().__init__()

        self.fourier = MPFourier(num_channels, bandwidth=bandwidth)
        self.final_linear = nn.Linear(num_channels*in_features, out_features)

    def forward(self, coords):
        coords = self.fourier(coords.unsqueeze(-1))
        coords = coords.flatten(-2)
        return self.final_linear(coords)

# MPFourier is from NVlabs/edm2 (Karras et al., 2024):
# https://github.com/NVlabs/edm2/blob/4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc/training/networks_edm2.py#L73
class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y * self.freqs.to(torch.float32)
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = getattr(F, config.hidden_act) if hasattr(F, config.hidden_act) else F.silu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class Transition(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size // 2
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = getattr(F, config.hidden_act) if hasattr(F, config.hidden_act) else F.silu

    def forward(self, x):
        res = x
        x = self.norm(x)
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj + res

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    cos and sin are [batch, seq, head_dim] and are unsqueezed at unsqueeze_dim to
    broadcast against q and k: 1 for [batch, heads, seq, head_dim], 2 for
    [batch, seq, heads, head_dim]. position_ids is unused. Returns (q, k) rotated.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()

        self.config = config

        base = config.rope_theta
        partial_rotary_factor = 1.0
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        self.attention_scaling = 1.0

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, position_ids, device, dtype):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(device)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=dtype), sin.to(dtype=dtype)

class StaticKV:
    """Decode-time KV cache for one attention layer, managed by a shared CoordCache.

    In commit mode, update() appends keys and values to the prefix. Otherwise it
    writes the query's keys into the scratch slot after the prefix and returns the
    cache's current view, masked by cache.valid. All diffusion steps of a token then
    have static shapes and can run as one CUDA graph.
    """
    def __init__(self, cache):
        self.cache = cache
        self.k = self.v = None
        self.len = 0

    def update(self, k, v):
        cache = self.cache
        if not cache.commit:
            self.k.index_copy_(2, cache.slot, k)
            self.v.index_copy_(2, cache.slot, v)
            return self.k[:, :, :cache.view], self.v[:, :, :cache.view]
        end = self.len + k.shape[2]
        if self.k is None or end + 1 > self.k.shape[2]:  # +1: room for the scratch slot
            size = cache.allocation(max(end + 1, 2 * (0 if self.k is None else self.k.shape[2])))
            # Zeros, not empty: masked NaN garbage would still propagate, since NaN * 0 = NaN.
            grown_k = k.new_zeros(*k.shape[:2], size, k.shape[3])
            grown_v = v.new_zeros(*v.shape[:2], size, v.shape[3])
            if self.len:
                grown_k[:, :, :self.len] = self.k[:, :, :self.len]
                grown_v[:, :, :self.len] = self.v[:, :, :self.len]
            self.k, self.v = grown_k, grown_v
            cache.graphs.clear()  # Captured graphs point at the old buffers.
        self.k[:, :, self.len:end] = k
        self.v[:, :, self.len:end] = v
        self.len = end
        return self.k[:, :, :end], self.v[:, :, :end]


class CoordCache:
    """Decode state that generate() keeps across tokens for sample_coord.

    Holds the KV cache of the sigma=0 prefix, grown in buckets, and one CUDA graph of
    the sampler loop per attended-length bucket. key_ok (bsz, capacity) masks prompt
    padding out of the coordinate transformer, so a padded row attends as it would
    unpadded.
    """
    bucket = 256

    def __init__(self, capacity, key_ok=None):
        self.capacity = capacity
        self.key_ok = key_ok
        self.layers = None
        self.graphs = {}
        self.commit = True
        self.view = 0

    def allocation(self, positions):
        return min(self.capacity, -(-positions // self.bucket) * self.bucket)

    def ensure(self, n_layers, device):
        if self.layers is None:
            rows = 1 if self.key_ok is None else self.key_ok.shape[0]
            self.slot = torch.zeros(1, dtype=torch.long, device=device)
            self.positions = torch.arange(self.capacity, device=device).view(1, 1, 1, -1)
            self.valid = torch.zeros(rows, 1, 1, self.capacity, dtype=torch.bool, device=device)
            self.layers = [StaticKV(self) for _ in range(n_layers)]

    @property
    def len(self):
        return self.layers[0].len

    def prefix_mask(self, start, end, device):
        """Attention mask for encoding prefix positions start..end-1 on top of the cache."""
        if self.key_ok is None:
            # Unpadded: a full prefill is plain causal. A later chunk needs an explicit mask.
            return torch.ones(end - start, end, dtype=torch.bool, device=device).tril(start) if start else None
        query = torch.arange(start, end, device=device)[:, None]
        key = torch.arange(end, device=device)[None, :]
        # Pad queries attend only to themselves, so no row is fully masked.
        return ((key <= query) & self.key_ok[:, None, None, :end]) | (key == query)

    def begin_step(self, prefix_len):
        """Point the scratch slot after the prefix; returns the attended view length."""
        self.commit = False
        self.slot.fill_(prefix_len)
        self.valid.copy_(self.positions <= prefix_len)  # copy_ broadcasts over rows. out= would resize.
        if self.key_ok is not None:
            self.valid &= self.key_ok[:, None, None, :]
        self.view = self.allocation(prefix_len + 1)
        return self.view


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value=None,
        block_mask=None,
        attn_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if isinstance(past_key_value, StaticKV):
            key_states, value_states = past_key_value.update(key_states, value_states)
            if not past_key_value.cache.commit:
                attn_mask = past_key_value.cache.valid[..., :past_key_value.cache.view]
            present_key_value = past_key_value
        else:
            if past_key_value is not None:
                past_key, past_value = past_key_value
                key_states = torch.cat((past_key, key_states), dim=2)
                value_states = torch.cat((past_value, value_states), dim=2)
            present_key_value = (key_states, value_states)

        q = query_states
        k = key_states
        v = value_states

        q = q.to(dtype=v.dtype)
        k = k.to(dtype=v.dtype)

        if isinstance(past_key_value, StaticKV) and not past_key_value.cache.commit:
            # One query position: fold each KV head's query group into the query axis
            # so the cache is attended in place, not expanded for GQA every step.
            b, h, n, d = q.shape
            attn_output = F.scaled_dot_product_attention(
                q.reshape(b, k.shape[1], h // k.shape[1] * n, d), k, v,
                scale=self.scaling, attn_mask=attn_mask,
            ).reshape(b, h, n, d)
        elif cu_seqlens is not None and past_key_value is None:
            attn_output = varlen_attention(q, k, v, cu_seqlens, max_seqlen)
        elif block_mask is None:
            if attn_mask is None:
                # Equal lengths mean prefill (causal). Otherwise a decode step attends to all history.
                is_causal = (q.shape[-2] == k.shape[-2])
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    is_causal=is_causal,
                    scale=self.scaling,
                    enable_gqa=True,
                )
            else:
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    scale=self.scaling,
                    enable_gqa=True,
                    attn_mask=attn_mask,
                )
        else:
            attn_output = flex_attention(q, k, v, block_mask=block_mask, scale=self.scaling, enable_gqa=True)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, present_key_value

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = Qwen3RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class Qwen3DecoderLayerAdaLN(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cond,
        past_key_value=None,
        block_mask=None,
        attn_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
    ) -> torch.Tensor:

        res = self.adaLN_modulation(cond).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = res

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = modulate(hidden_states, shift_msa, scale_msa)

        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            block_mask=block_mask,
            attn_mask=attn_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        hidden_states = residual + hidden_states*gate_msa

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = modulate(hidden_states, shift_mlp, scale_mlp)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states*gate_mlp
        return hidden_states, present_key_value


class Qwen3AttentionMoT(nn.Module):
    """MoT attention with modality-specific q/k/v/o and mixed kv caches."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.text_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.xyz_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.text_attn.head_dim
        self.scaling = self.text_attn.scaling

    def _project(self, attn: Qwen3Attention, hidden_states: torch.Tensor):
        input_shape = hidden_states.shape[:-1]
        query_states = attn.q_norm(
            attn.q_proj(hidden_states).view(*input_shape, self.num_attention_heads, self.head_dim)
        )
        key_states = attn.k_norm(
            attn.k_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim)
        )
        value_states = attn.v_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim)
        return query_states, key_states, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        has_xyz=None,
        past_key_value=None,
        block_mask=None,
        attn_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
    ):
        input_shape = hidden_states.shape[:-1]
        if has_xyz is None:
            has_xyz = torch.zeros(input_shape, dtype=torch.bool, device=hidden_states.device)
        else:
            has_xyz = has_xyz.to(device=hidden_states.device, dtype=torch.bool).reshape(input_shape)
        modality_mask = has_xyz.unsqueeze(-1).unsqueeze(-1)

        text_q, text_k, text_v = self._project(self.text_attn, hidden_states)
        xyz_q, xyz_k, xyz_v = self._project(self.xyz_attn, hidden_states)

        query_states = torch.where(modality_mask, xyz_q, text_q)
        key_states = torch.where(modality_mask, xyz_k, text_k)
        value_states = torch.where(modality_mask, xyz_v, text_v)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat((past_key, key_states), dim=2)
            value_states = torch.cat((past_value, value_states), dim=2)

        present_key_value = (key_states, value_states)

        q = query_states.to(dtype=value_states.dtype)
        k = key_states.to(dtype=value_states.dtype)
        v = value_states

        if cu_seqlens is not None and past_key_value is None:
            attn_output = varlen_attention(q, k, v, cu_seqlens, max_seqlen)
        elif block_mask is None:
            if attn_mask is None:
                is_causal = q.shape[-2] == k.shape[-2]
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    is_causal=is_causal,
                    scale=self.scaling,
                    enable_gqa=True,
                )
            else:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    scale=self.scaling,
                    enable_gqa=True,
                    attn_mask=attn_mask,
                )
        else:
            attn_output = flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                scale=self.scaling,
                enable_gqa=True,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        text_out = self.text_attn.o_proj(attn_output)
        xyz_out = self.xyz_attn.o_proj(attn_output)
        hidden_states = torch.where(has_xyz.unsqueeze(-1), xyz_out, text_out)
        return hidden_states, present_key_value


class Qwen3DecoderLayerMoT(nn.Module):
    """Full MoT layer with modality-specific attention and FFN."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.input_layernorm_text = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_xyz = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen3AttentionMoT(config=config, layer_idx=layer_idx)

        self.post_attention_layernorm_text = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_text = Qwen3MLP(config)

        self.post_attention_layernorm_xyz = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_xyz = Qwen3MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        has_xyz=None,
        past_key_value=None,
        block_mask=None,
        attn_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
    ):
        input_shape = hidden_states.shape[:-1]
        if has_xyz is None:
            has_xyz = torch.zeros(input_shape, dtype=torch.bool, device=hidden_states.device)
        else:
            has_xyz = has_xyz.to(device=hidden_states.device, dtype=torch.bool).reshape(input_shape)

        residual = hidden_states
        text_in = self.input_layernorm_text(hidden_states)
        xyz_in = self.input_layernorm_xyz(hidden_states)
        routed_hidden_states = torch.where(has_xyz.unsqueeze(-1), xyz_in, text_in)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=routed_hidden_states,
            position_embeddings=position_embeddings,
            has_xyz=has_xyz,
            past_key_value=past_key_value,
            block_mask=block_mask,
            attn_mask=attn_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        text_out = self.mlp_text(self.post_attention_layernorm_text(hidden_states))
        xyz_out = self.mlp_xyz(self.post_attention_layernorm_xyz(hidden_states))
        routed_hidden_states = torch.where(has_xyz.unsqueeze(-1), xyz_out, text_out)
        hidden_states = residual + routed_hidden_states
        return hidden_states, present_key_value


class NextTokenDiffusion(nn.Module):
    """Full MoT trunk with a causal, per-token AdaLN transformer denoiser."""


    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config

        self.true_vocab_size = config.lm_vocab_size + config.atom_vocab_size

        self.embed_tokens = nn.Embedding(config.lm_vocab_size, config.hidden_size)
        self.embed_atoms = nn.Embedding(config.atom_vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            Qwen3DecoderLayerMoT(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        self.embed_coords = nn.Linear(3, config.hidden_size, bias=True)

        self.cond_stem = nn.Sequential(
            Transition(config),
            Transition(config),
        )

        self.layers2 = nn.ModuleList(
            [Qwen3DecoderLayerAdaLN(config, layer_idx) for layer_idx in range(config.num_coord_transformer_layers)]
        )

        self.embed_fourier = FourierCoords(
            in_features=3,
            num_channels=config.coord_fourier,
            out_features=config.hidden_size,
            bandwidth=config.coord_bandwidth,
        )

        self.lm_head = nn.Linear(config.hidden_size, config.lm_vocab_size, bias=False)
        self.atom_head = nn.Linear(config.hidden_size, config.atom_vocab_size, bias=False)

        self.diff_head = FinalLayer(config.hidden_size, 3)

        # Attention backend for the trunk and the coordinate transformer during training.
        self.attention_backend = resolve_attention_backend(getattr(config, "attention_backend", "flex"))

        self.time_embed = MPFourier(config.hidden_size)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        for block in self.layers2:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.diff_head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.diff_head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.diff_head.linear.weight, 0)
        nn.init.constant_(self.diff_head.linear.bias, 0)


    def freeze_llm_weights(self):
        """Freeze pretrained text branch, leaving xyz-specific params trainable."""
        self.embed_tokens.weight.requires_grad = False
        self.lm_head.weight.requires_grad = False
        for param in self.norm.parameters():
            param.requires_grad = False
        for layer in self.layers:
            for param in layer.input_layernorm_text.parameters():
                param.requires_grad = False
            for param in layer.self_attn.text_attn.parameters():
                param.requires_grad = False
            for param in layer.mlp_text.parameters():
                param.requires_grad = False
            for param in layer.post_attention_layernorm_text.parameters():
                param.requires_grad = False


    def load_qwen3_weights(self, model_dir, init_xyz_from_pretrained: bool = True):
        """Load Qwen3 weights into the text branch and optionally mirror them to xyz."""
        index_file = model_dir / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file, "r", encoding="utf-8") as f:
                weight_map = json.load(f)["weight_map"]
            files = sorted({model_dir / fname for fname in weight_map.values()})
        else:
            files = list(model_dir.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(f"No *.safetensors files found in {model_dir}")

        tensors = {}
        for fpath in files:
            with safe_open(fpath, framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    tensors[key] = sf.get_tensor(key)

        def copy_param(param: nn.Parameter, key: str):
            param.copy_(tensors[key].to(dtype=param.dtype, device=param.device))

        with torch.no_grad():
            emb_key = "model.embed_tokens.weight"
            self.embed_tokens.weight.copy_(
                tensors[emb_key][:self.config.lm_vocab_size].to(
                    dtype=self.embed_tokens.weight.dtype,
                    device=self.embed_tokens.weight.device,
                )
            )
            self.lm_head.weight.copy_(
                tensors[emb_key][:self.config.lm_vocab_size].to(
                    dtype=self.lm_head.weight.dtype,
                    device=self.lm_head.weight.device,
                )
            )

            for i, layer in enumerate(self.layers):
                prefix = f"model.layers.{i}."

                copy_param(layer.input_layernorm_text.weight, prefix + "input_layernorm.weight")
                if init_xyz_from_pretrained:
                    copy_param(layer.input_layernorm_xyz.weight, prefix + "input_layernorm.weight")

                for branch_name, should_init in (("text_attn", True), ("xyz_attn", init_xyz_from_pretrained)):
                    if not should_init:
                        continue
                    branch = getattr(layer.self_attn, branch_name)
                    for param_name, param in branch.named_parameters():
                        copy_param(param, prefix + "self_attn." + param_name)

                copy_param(layer.post_attention_layernorm_text.weight, prefix + "post_attention_layernorm.weight")
                if init_xyz_from_pretrained:
                    copy_param(layer.post_attention_layernorm_xyz.weight, prefix + "post_attention_layernorm.weight")

                for branch_name, should_init in (("mlp_text", True), ("mlp_xyz", init_xyz_from_pretrained)):
                    if not should_init:
                        continue
                    branch = getattr(layer, branch_name)
                    for param_name, param in branch.named_parameters():
                        copy_param(param, prefix + "mlp." + param_name)

            copy_param(self.norm.weight, "model.norm.weight")

        tensors.clear()


    def encode1(self, position_embeddings, atoms, coords, past_key_values=None, block_mask=None, attn_mask=None,
                cu_seqlens=None, max_seqlen=None):
        # Tokens that carry floats (atoms, lattice, <1> slots) use the xyz branch.
        # Discrete tokens (text, <scalar>, </scalar>) use the text branch.
        has_xyz = has_coords(atoms)
        embedding = torch.cat([self.embed_tokens.weight, self.embed_atoms.weight], dim=0)
        seq = F.embedding(atoms, embedding)

        out = self.embed_coords(coords)
        out = torch.where(has_xyz.unsqueeze(-1), out, torch.zeros_like(out))
        seq = seq + out

        out = self.embed_fourier(coords)
        out = torch.where(has_xyz.unsqueeze(-1), out, torch.zeros_like(out))
        seq = seq + out

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        new_key_values = []
        for i, block in enumerate(self.layers):
            seq, layer_past = block(
                seq, position_embeddings, has_xyz=has_xyz,
                past_key_value=past_key_values[i], block_mask=block_mask, attn_mask=attn_mask,
                cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            new_key_values.append(layer_past)

        seq = self.norm(seq)
        return seq, new_key_values


    def make_cond(self, target_atoms, h1, t_emb, base=None):
        """base: optional precomputed embedding(target_atoms) + h1, which is constant
        across a token's diffusion steps."""
        if base is None:
            embedding = torch.cat([self.embed_tokens.weight, self.embed_atoms.weight], dim=0)
            base = F.embedding(target_atoms, embedding) + h1
        cond = base + t_emb
        cond = self.cond_stem(cond)
        return cond


    def encode2(self, x_noisy, cond, position_embeddings, past_key_values=None, block_mask=None,
                attn_mask=None, cu_seqlens=None, max_seqlen=None):
        seq = self.embed_coords(x_noisy)
        seq += self.embed_fourier(x_noisy)
        seq += cond

        if past_key_values is None:
            past_key_values = [None] * len(self.layers2)

        new_key_values = []

        for i, block in enumerate(self.layers2):
            seq, layer_past = block(seq, position_embeddings, cond=cond,
                                    past_key_value=past_key_values[i], block_mask=block_mask,
                                    attn_mask=attn_mask, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            new_key_values.append(layer_past)

        return seq, new_key_values


    def denoise(self, x_noisy, sigma, target_atoms, h1, position_embeddings, past_key_values=None,
                block_mask=None, attn_mask=None, cu_seqlens=None, max_seqlen=None, cond_base=None):
        sigma_data = self.config.sigma_data

        c_in = 1 / (sigma_data ** 2 + sigma ** 2).sqrt()
        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()

        t_emb = self.time_embed(sigma).reshape(h1.shape)
        cond = self.make_cond(target_atoms, h1, t_emb, base=cond_base)

        h2, _ = self.encode2(c_in * x_noisy, cond, position_embeddings, past_key_values=past_key_values,
                             block_mask=block_mask, attn_mask=attn_mask,
                             cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        F_x = self.diff_head(h2, cond)

        return c_skip * x_noisy + c_out * F_x


    def forward(self, idx, atoms, coords, target_atoms, target_coords, document_ids, target_loss_mask):
        # Masks are built outside the compiled region because torch.nonzero has a
        # data-dependent shape. mark_dynamic avoids one recompile per document count.
        block_mask = None
        cu_seqlens = None
        if getattr(self, "attention_backend", "flex") == "varlen":
            cu_seqlens = create_cu_seqlens(document_ids)
            torch._dynamo.mark_dynamic(cu_seqlens, 0)
        else:
            block_mask = create_document_block_mask(document_ids)

        if getattr(self, "_compiled_impl", None) is None:
            self._compiled_impl = (torch.compile(self._forward_impl, dynamic=False, fullgraph=True)
                                   if self.config.compile else self._forward_impl)
        return self._compiled_impl(
            idx, atoms, coords, target_atoms, target_coords, document_ids,
            target_loss_mask, block_mask=block_mask, cu_seqlens=cu_seqlens,
        )


    def _forward_impl(self, idx, atoms, coords, target_atoms, target_coords, document_ids,
                      target_loss_mask, block_mask=None, cu_seqlens=None):
        max_seqlen = int(getattr(self.config, "max_len", 4096))
        idx = idx.unsqueeze(0)
        atoms = atoms.unsqueeze(0)
        coords = coords.unsqueeze(0)
        target_atoms = target_atoms.unsqueeze(0)
        target_coords = target_coords.unsqueeze(0)
        target_loss_mask = target_loss_mask.unsqueeze(0)

        device = coords.device
        dtype = self.embed_tokens.weight.dtype
        position_embeddings = self.rotary_emb(idx, device=device, dtype=dtype)

        seq, _ = self.encode1(position_embeddings, atoms, coords, block_mask=block_mask,
                              cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        lm_logits = self.lm_head(seq)
        atom_logits = self.atom_head(seq.clone())
        logits = torch.cat([lm_logits, atom_logits], dim=-1)
        ce_targets = target_atoms
        if self.config.response_loss_only:
            ce_targets = ce_targets.masked_fill(~target_loss_mask.bool(), -100)
        cross_entropy = F.cross_entropy(
            logits.view(-1, self.true_vocab_size),
            ce_targets.reshape(-1),
            ignore_index=-100, reduction="sum",
        ) / (ce_targets != -100).sum().clamp_min(1)

        # timestep sampling
        bsz, L, _ = target_coords.shape
        rnd_normal = torch.randn([bsz, L, 1], device=device)
        sigma_data = self.config.sigma_data
        P_mean = self.config.P_mean
        P_std = self.config.P_std
        sigma = (rnd_normal * P_std + P_mean).exp()
        weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2

        # Zero inactive float components before noising. A <1> scalar uses entry 0,
        # an atom all 3.
        comp_mask = coord_component_mask(target_atoms, n_components=target_coords.shape[-1]).to(dtype=target_coords.dtype)
        target_coords = target_coords * comp_mask
        noisy_coords = (target_coords + torch.randn_like(target_coords) * sigma) * comp_mask

        pred_coords = self.denoise(noisy_coords, sigma, target_atoms, seq, position_embeddings,
                                   block_mask=block_mask, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        loss = weight * ((pred_coords - target_coords) ** 2)
        xyz_mask = has_coords(target_atoms)
        if self.config.response_loss_only:
            xyz_mask = xyz_mask & target_loss_mask.bool()
        mask = (xyz_mask.unsqueeze(-1) & comp_mask.bool()).to(dtype=loss.dtype)
        loss = loss * mask
        if self.config.response_loss_only:
            denom = mask.sum().clamp_min(1.0)
            loss = loss.sum() / denom
        else:
            loss = loss.mean()
        return loss, cross_entropy


    def _heun(self, x_next, t_steps, comp_keep, target_atoms_cur, h1_cur, pos_emb_cur,
              prefix_cache, denoise_mask=None, intermediates=None):
        """EDM Heun sampler for the current token. Static shapes and no host syncs,
        so it can be captured as a CUDA graph."""
        bsz = x_next.shape[0]
        num_steps = t_steps.shape[0] - 1
        embedding = torch.cat([self.embed_tokens.weight, self.embed_atoms.weight], dim=0)
        base = F.embedding(target_atoms_cur, embedding) + h1_cur
        for i in range(num_steps):
            t_cur, t_next = t_steps[i], t_steps[i + 1]
            denoised = self.denoise(x_next, t_cur.expand(bsz, 1, 1), target_atoms_cur, h1_cur, pos_emb_cur,
                                    past_key_values=prefix_cache, attn_mask=denoise_mask, cond_base=base)
            d_cur = (x_next - denoised) / t_cur
            x_prime = (x_next + (t_next - t_cur) * d_cur) * comp_keep
            if i == num_steps - 1:  # t_next == 0
                x_next = x_prime
            else:
                denoised_next = self.denoise(x_prime, t_next.expand(bsz, 1, 1), target_atoms_cur, h1_cur,
                                             pos_emb_cur, past_key_values=prefix_cache, attn_mask=denoise_mask,
                                             cond_base=base)
                d_prime = (x_prime - denoised_next) / t_next
                x_next = (x_next + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)) * comp_keep
            if intermediates is not None:
                intermediates.append(x_next.clone())
        return x_next

    @torch.no_grad()
    def sample_coord(self, atoms_seq, coords_seq, target_atoms_seq, h1, position_embeddings,
                     num_steps=18, sigma_min=1e-4, sigma_max=80, rho=7, attn_mask=None, return_intermediates=False,
                     coord_cache=None):
        """Sample the current token's coordinates given the generated prefix.

        With coord_cache (a CoordCache from generate()), prefix positions are encoded at
        sigma=0 under a causal mask. Their keys and values never change, so only new
        positions are encoded and the sampler loop replays as a CUDA graph.
        """
        bsz = h1.shape[0]
        device = h1.device

        step_indices = torch.arange(num_steps, device=device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        x_next = torch.randn(bsz, 1, 3, device=device) * t_steps[0]

        past_target_coords = coords_seq[:, 1:] if coords_seq.shape[1] > 1 else torch.zeros(bsz, 0, 3, device=device)
        prefix_len = past_target_coords.shape[1]

        # Prepare target_mask
        target_mask = None
        if attn_mask is not None:
            if attn_mask.shape[-1] > target_atoms_seq.shape[1]:
                target_mask = attn_mask[:, :, 1:, 1:]

        # Inactive float components stay zero through the sampler loop.
        comp_keep = coord_component_mask(target_atoms_seq[:, prefix_len:], n_components=3).to(x_next.dtype)
        x_next = x_next * comp_keep
        target_atoms_cur = target_atoms_seq[:, prefix_len:]
        h1_cur = h1[:, prefix_len:]
        pos_emb_cur = (position_embeddings[0][:, prefix_len:], position_embeddings[1][:, prefix_len:])

        if coord_cache is not None and target_mask is None and not return_intermediates:
            return self._sample_coord_cached(coord_cache, x_next, t_steps, comp_keep, past_target_coords,
                                             target_atoms_seq, h1, position_embeddings).squeeze(1)

        # KV cache of the clean (sigma=0) prefix.
        prefix_cache = None

        if prefix_len > 0:
            sigma_past = torch.zeros(bsz, prefix_len, 1, device=device)

            target_atoms_prefix = target_atoms_seq[:, :prefix_len]
            h1_prefix = h1[:, :prefix_len]

            t_emb_prefix = self.time_embed(sigma_past.reshape(-1, 1)).reshape(bsz, prefix_len, -1)
            cond_prefix = self.make_cond(target_atoms_prefix, h1_prefix, t_emb_prefix)

            sigma_data = self.config.sigma_data
            c_in_prefix = 1.0 / sigma_data

            x_in_prefix = c_in_prefix * past_target_coords

            pos_emb_prefix = (position_embeddings[0][:, :prefix_len], position_embeddings[1][:, :prefix_len])

            prefix_mask = None
            if target_mask is not None:
                prefix_mask = target_mask[:, :, :prefix_len, :prefix_len]

            _, prefix_cache = self.encode2(
                x_in_prefix,
                cond_prefix,
                pos_emb_prefix,
                past_key_values=None,
                attn_mask=prefix_mask
            )

        denoise_mask = None
        if target_mask is not None:
            denoise_mask = target_mask[:, :, prefix_len:, :prefix_len + 1]

        intermediates = [x_next.clone()] if return_intermediates else None
        x_next = self._heun(x_next, t_steps, comp_keep, target_atoms_cur, h1_cur, pos_emb_cur,
                            prefix_cache, denoise_mask, intermediates)

        if return_intermediates:
            return x_next.squeeze(1), torch.stack(intermediates, dim=1)
        return x_next.squeeze(1)

    def _sample_coord_cached(self, cache, x_next, t_steps, comp_keep, past_target_coords,
                             target_atoms_seq, h1, position_embeddings):
        bsz, device = h1.shape[0], h1.device
        prefix_len = past_target_coords.shape[1]
        cache.ensure(len(self.layers2), device)

        start = cache.len
        if prefix_len > start:
            new = prefix_len - start
            t_emb = self.time_embed(torch.zeros(bsz * new, 1, device=device)).reshape(bsz, new, -1)
            cond = self.make_cond(target_atoms_seq[:, start:prefix_len], h1[:, start:prefix_len], t_emb)
            pos = (position_embeddings[0][:, start:prefix_len], position_embeddings[1][:, start:prefix_len])
            cache.commit = True
            self.encode2((1.0 / self.config.sigma_data) * past_target_coords[:, start:], cond, pos,
                         past_key_values=cache.layers, attn_mask=cache.prefix_mask(start, prefix_len, device))

        # The current token's queries go in the slot after the prefix, attending to keys 0..prefix_len.
        view = cache.begin_step(prefix_len)
        inputs = dict(x=x_next, keep=comp_keep, atoms=target_atoms_seq[:, prefix_len:],
                      h1=h1[:, prefix_len:], cos=position_embeddings[0][:, prefix_len:],
                      sin=position_embeddings[1][:, prefix_len:])

        def run():
            io = cache.inputs
            return self._heun(io['x'], cache.t_steps, io['keep'], io['atoms'], io['h1'],
                              (io['cos'], io['sin']), cache.layers)

        if not h1.is_cuda:
            cache.inputs, cache.t_steps = inputs, t_steps
            return run()
        if not cache.graphs:
            cache.inputs = {k: v.clone() for k, v in inputs.items()}
            cache.t_steps = t_steps
        else:
            for k, v in inputs.items():
                cache.inputs[k].copy_(v)
        if view not in cache.graphs:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = run()
            cache.graphs[view] = (graph, output)
        graph, output = cache.graphs[view]
        graph.replay()
        return output.clone()


    @torch.no_grad()
    def generate(self, prefix=None, bsz=1, max_len=32, pbar=True, stop_token=STOP, keep_prefix=True, temperature=1.0, atom_logit_shift=0, num_steps=18, pad_mask=None, return_intermediates=False, forced_tokens=None, forced_lens=None, **kwargs):
        """Sample tokens and coordinates autoregressively. Returns (tokens, coords) on CPU.

        pad_mask marks real prompt tokens. Padding is masked in both the trunk and the
        coordinate transformer, so a padded row is sampled exactly as it would be alone."""

        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype

        # Initialize from prefix or from scratch
        if pad_mask is not None:
            if isinstance(pad_mask, np.ndarray):
                pad_mask = torch.tensor(pad_mask, dtype=torch.bool, device=device)

        if forced_tokens is not None:
            if isinstance(forced_tokens, np.ndarray):
                forced_tokens = torch.tensor(forced_tokens, dtype=torch.long, device=device)
            forced_tokens = forced_tokens.to(device=device, dtype=torch.long)
            if forced_lens is None:
                raise ValueError("forced_lens must be provided alongside forced_tokens")
            if isinstance(forced_lens, np.ndarray):
                forced_lens = torch.tensor(forced_lens, dtype=torch.long, device=device)
            forced_lens = forced_lens.to(device=device, dtype=torch.long)

        if prefix is None:
            tokens = torch.full((bsz, 1), GEN, dtype=torch.long, device=device)
            coords = torch.zeros(bsz, 1, 3, device=device)
        else:
            tokens, coords = prefix
            if isinstance(tokens, np.ndarray):
                tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            if isinstance(coords, np.ndarray):
                coords = torch.tensor(coords, dtype=torch.float32, device=device)

            tokens = tokens.to(device)
            if coords is None:
                coords = torch.zeros(*tokens.shape, 3, device=device)
            else:
                coords = coords.to(device, dtype=torch.float32)

            # Zero each prefix token's inactive float components.
            coords = coords * coord_component_mask(tokens, n_components=coords.shape[-1]).to(coords.dtype)

            bsz = tokens.shape[0]
            max_len = max_len - tokens.shape[1]

        prefix_len = tokens.shape[1]
        stop_mask = torch.zeros(bsz, dtype=torch.bool, device=device)
        past_key_values = None
        h1_cache = None

        intermediates_list = []

        pbar_iter = tqdm.trange(max_len) if pbar else range(max_len)
        capacity = tokens.shape[1] + max_len + 1
        key_ok = None
        if pad_mask is not None and not bool(pad_mask.all()):
            # Coordinate position p pairs trunk state p with target token p + 1, so it is
            # real only if both tokens are. The first real token is then never a key,
            # as in the unpadded case.
            key_ok = torch.ones(bsz, capacity, dtype=torch.bool, device=device)
            key_ok[:, :pad_mask.shape[1] - 1] = pad_mask[:, :-1] & pad_mask[:, 1:]
        coord_cache = CoordCache(capacity, key_ok)
        for i in pbar_iter:
            bsz, seq_len = tokens.shape
            if pad_mask is not None:
                if past_key_values is None:
                    lower_tri = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
                    attn_mask = pad_mask[:, None, None, :] & lower_tri[None, :, :]
                else:  # The last row of the causal mask is all ones.
                    attn_mask = pad_mask[:, None, None, :]
                idx = (pad_mask.cumsum(dim=-1) - 1).clamp(min=0)
            else:
                attn_mask = None
                idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
            position_embeddings = self.rotary_emb(idx, device=device, dtype=dtype)

            if past_key_values is None:
                atoms_in = tokens
                coords_in = coords
                pos_emb_in = position_embeddings
            else:
                atoms_in = tokens[:, -1:]
                coords_in = coords[:, -1:]
                pos_emb_in = (position_embeddings[0][:, -1:], position_embeddings[1][:, -1:])
                if attn_mask is not None and attn_mask.shape[-2] > 1:
                    attn_mask = attn_mask[:, :, -1:, :]

            seq, past_key_values = self.encode1(pos_emb_in, atoms_in, coords_in, past_key_values=past_key_values, attn_mask=attn_mask)

            if h1_cache is None:
                h1_cache = seq
            else:
                h1_cache = torch.cat([h1_cache, seq], dim=1)

            lm_logits = self.lm_head(seq[:, -1, :])
            atom_logits = self.atom_head(seq[:, -1, :]) + atom_logit_shift
            logits = torch.cat([lm_logits, atom_logits], dim=-1)

            if temperature <= 0.0:
                next_atom = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_atom = torch.multinomial(probs, num_samples=1)

            if forced_tokens is not None and i < forced_tokens.shape[1]:
                do_force = (forced_lens > i).unsqueeze(-1)
                forced_now = forced_tokens[:, i:i+1]
                next_atom = torch.where(do_force, forced_now, next_atom)

            # Finished rows keep emitting the stop token so they never trigger the
            # coordinate sampler.
            next_atom = torch.where(stop_mask.unsqueeze(-1), torch.full_like(next_atom, stop_token), next_atom)
            stop_mask |= next_atom.squeeze(-1) == stop_token
            if stop_mask.all():
                break

            tokens = torch.cat([tokens, next_atom], dim=1)

            target_atoms_seq = tokens[:, 1:]

            # Run the coordinate sampler only if some row's new token carries floats.
            # Other tokens get zero coordinates.
            next_has_coords = has_coords(next_atom)  # (bsz, 1)
            if bool(next_has_coords.any()):
                sample_out = self.sample_coord(
                    atoms_seq=tokens[:, :-1],
                    coords_seq=coords,
                    target_atoms_seq=target_atoms_seq,
                    h1=h1_cache,
                    position_embeddings=position_embeddings,
                    num_steps=num_steps,
                    attn_mask=attn_mask,
                    return_intermediates=return_intermediates,
                    coord_cache=coord_cache,
                    **kwargs
                )
                if return_intermediates:
                    next_coord, intermediates = sample_out
                    intermediates_list.append(intermediates)
                else:
                    next_coord = sample_out
            else:
                next_coord = torch.zeros(bsz, 3, device=device, dtype=coords.dtype)
                if return_intermediates:
                    intermediates_list.append(torch.zeros(bsz, 1, 3, device=device, dtype=coords.dtype))

            # Zero the new token's inactive float components.
            comp_keep = coord_component_mask(next_atom, n_components=next_coord.shape[-1]).reshape(next_coord.shape)
            next_coord = next_coord * comp_keep.to(next_coord.dtype)

            # Append the sampled coordinate
            coords = torch.cat([coords, next_coord.unsqueeze(1)], dim=1)

            if pad_mask is not None:
                pad_mask = torch.cat([pad_mask, torch.ones((bsz, 1), dtype=torch.bool, device=device)], dim=1)

        # Free the graphs now. StaticKV and CoordCache form a reference cycle, and graphs
        # freed by the garbage collector during a later graph capture invalidate it.
        coord_cache.graphs.clear()

        if not keep_prefix:
            tokens = tokens[:, prefix_len:]
            coords = coords[:, prefix_len:]

        tokens = tokens.cpu()
        coords = coords.cpu()
        if return_intermediates:
            return tokens, coords, intermediates_list
        return tokens, coords


class ResBlock(nn.Module):
    """AdaLN residual MLP block from MAR's Diffusion Loss head."""

    def __init__(self, channels, expand=1):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, expand * channels, bias=True),
            nn.SiLU(),
            nn.Linear(expand * channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True),
        )

    def forward(self, x, y):
        shift, scale, gate = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift, scale)
        h = self.mlp(h)
        return x + gate * h


class DiffMLPFinalLayer(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 2 * channels, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiffMLP(nn.Module):
    """Per-token diffusion head: input proj + Fourier + AdaLN ResBlocks. No cross-token mixing."""

    def __init__(self, config, channels=3):
        super().__init__()
        diff_w = config.diff_w
        diff_d = config.diff_d
        diff_fourier = config.coord_fourier
        expand = config.diff_mlp_expand

        self.input_proj = nn.Linear(channels, diff_w)
        self.embed_fourier = FourierCoords(
            in_features=channels,
            num_channels=diff_fourier,
            out_features=diff_w,
            bandwidth=config.coord_bandwidth,
        )
        self.cond_embed = nn.Linear(config.hidden_size, diff_w)

        self.res_blocks = nn.ModuleList([ResBlock(diff_w, expand=expand) for _ in range(diff_d)])
        self.final_layer = DiffMLPFinalLayer(diff_w, channels)

        self._initialize_weights()

    def _initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, cond):
        h = self.input_proj(x) + self.embed_fourier(x)
        c = self.cond_embed(cond)
        for block in self.res_blocks:
            h = block(h, c)
        return self.final_layer(h, c)


class DiffusionMLP(NextTokenDiffusion):
    """MoT Nahual with a per-token DiffMLP denoiser instead of the coord transformer."""

    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config


        self.true_vocab_size = config.lm_vocab_size + config.atom_vocab_size

        self.embed_tokens = nn.Embedding(config.lm_vocab_size, config.hidden_size)
        self.embed_atoms = nn.Embedding(config.atom_vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            Qwen3DecoderLayerMoT(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        self.embed_coords = nn.Linear(3, config.hidden_size, bias=True)

        self.cond_stem = nn.Sequential(
            Transition(config),
            Transition(config),
        )

        self.embed_fourier = FourierCoords(
            in_features=3,
            num_channels=config.coord_fourier,
            out_features=config.hidden_size,
            bandwidth=config.coord_bandwidth,
        )

        self.lm_head = nn.Linear(config.hidden_size, config.lm_vocab_size, bias=False)
        self.atom_head = nn.Linear(config.hidden_size, config.atom_vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.time_embed = MPFourier(config.hidden_size)

        self.diff_mlp = DiffMLP(config, channels=3)

        # Trunk attention backend. The DiffMLP denoiser has no attention.
        self.attention_backend = resolve_attention_backend(getattr(config, "attention_backend", "flex"))
        self.max_document_size = int(getattr(config, "max_len", 4096))
        self._compiled_impl = None

    def denoise(
        self,
        x_noisy,
        sigma,
        target_atoms,
        h1,
        position_embeddings=None,
        past_key_values=None,
        block_mask=None,
        attn_mask=None,
    ):
        del position_embeddings, past_key_values, block_mask, attn_mask
        sigma_data = self.config.sigma_data
        c_in = 1.0 / (sigma_data ** 2 + sigma ** 2).sqrt()
        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()

        t_emb = self.time_embed(sigma).reshape(h1.shape)
        cond = self.make_cond(target_atoms, h1, t_emb)

        F_x = self.diff_mlp(c_in * x_noisy, cond)
        return c_skip * x_noisy + c_out * F_x

    def _forward_impl(self, idx, atoms, coords, target_atoms, target_coords, document_ids,
                      target_loss_mask, block_mask=None, cu_seqlens=None):
        idx = idx.unsqueeze(0)
        atoms = atoms.unsqueeze(0)
        coords = coords.unsqueeze(0)
        target_atoms = target_atoms.unsqueeze(0)
        target_coords = target_coords.unsqueeze(0)
        target_loss_mask = target_loss_mask.unsqueeze(0)

        device = coords.device
        dtype = self.embed_tokens.weight.dtype
        position_embeddings = self.rotary_emb(idx, device=device, dtype=dtype)

        seq, _ = self.encode1(
            position_embeddings, atoms, coords, block_mask=block_mask,
            cu_seqlens=cu_seqlens, max_seqlen=self.max_document_size,
        )
        lm_logits = self.lm_head(seq)
        atom_logits = self.atom_head(seq.clone())
        logits = torch.cat([lm_logits, atom_logits], dim=-1)
        ce_targets = target_atoms
        if self.config.response_loss_only:
            ce_targets = ce_targets.masked_fill(~target_loss_mask.bool(), -100)
        cross_entropy = F.cross_entropy(
            logits.view(-1, self.true_vocab_size),
            ce_targets.reshape(-1),
            ignore_index=-100, reduction="sum",
        ) / (ce_targets != -100).sum().clamp_min(1)

        bsz, L, _ = target_coords.shape
        M = self.config.diff_mult

        target_coords_rep = target_coords.repeat(M, 1, 1)
        target_atoms_rep = target_atoms.repeat(M, 1)
        seq_rep = seq.repeat(M, 1, 1)

        sigma_data = self.config.sigma_data
        P_mean = self.config.P_mean
        P_std = self.config.P_std
        rnd_normal = torch.randn([bsz * M, L, 1], device=device)
        sigma = (rnd_normal * P_std + P_mean).exp()
        weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2

        # Zero inactive float components before noising.
        comp_mask = coord_component_mask(target_atoms_rep, n_components=target_coords_rep.shape[-1]).to(dtype=target_coords_rep.dtype)
        target_coords_rep = target_coords_rep * comp_mask
        noisy_coords = (target_coords_rep + torch.randn_like(target_coords_rep) * sigma) * comp_mask
        pred_coords = self.denoise(noisy_coords, sigma, target_atoms_rep, seq_rep)

        loss = weight * ((pred_coords - target_coords_rep) ** 2)
        xyz_mask = has_coords(target_atoms_rep)
        if self.config.response_loss_only:
            tlm_rep = target_loss_mask.repeat(M, 1).bool()
            xyz_mask = xyz_mask & tlm_rep
        mask = (xyz_mask.unsqueeze(-1) & comp_mask.bool()).to(dtype=loss.dtype)
        loss = loss * mask
        if self.config.response_loss_only:
            denom = mask.sum().clamp_min(1.0)
            loss = loss.sum() / denom
        else:
            loss = loss.mean()
        return loss, cross_entropy

    def forward(self, idx, atoms, coords, target_atoms, target_coords, document_ids, target_loss_mask):
        # Masks are built outside the compiled region because torch.nonzero has a
        # data-dependent shape. mark_dynamic avoids one recompile per document count.
        block_mask = None
        cu_seqlens = None
        if self.attention_backend == "varlen":
            cu_seqlens = create_cu_seqlens(document_ids)
            torch._dynamo.mark_dynamic(cu_seqlens, 0)
        else:
            block_mask = create_document_block_mask(document_ids)

        if self._compiled_impl is None:
            self._compiled_impl = (torch.compile(self._forward_impl, dynamic=False, fullgraph=True)
                                   if self.config.compile else self._forward_impl)
        return self._compiled_impl(
            idx, atoms, coords, target_atoms, target_coords, document_ids,
            target_loss_mask, block_mask=block_mask, cu_seqlens=cu_seqlens,
        )

    @torch.no_grad()
    def sample_coord(
        self,
        atoms_seq,
        coords_seq,
        target_atoms_seq,
        h1,
        position_embeddings,
        num_steps=18,
        sigma_min=1e-4,
        sigma_max=80,
        rho=7,
        attn_mask=None,
        return_intermediates=False,
        coord_cache=None,
    ):
        del atoms_seq, coords_seq, position_embeddings, attn_mask, coord_cache
        bsz = h1.shape[0]
        device = h1.device
        dtype = h1.dtype

        target_atoms_cur = target_atoms_seq[:, -1:]
        h1_cur = h1[:, -1:]

        step_indices = torch.arange(num_steps, device=device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        # Only the token's active float components evolve. The rest stay zero, as in training.
        comp_keep = coord_component_mask(target_atoms_cur, n_components=3).to(dtype)
        x_next = torch.randn(bsz, 1, 3, device=device, dtype=dtype) * t_steps[0] * comp_keep

        intermediates = []
        if return_intermediates:
            intermediates.append(x_next.clone())

        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            sigma_cur = t_cur.expand(bsz, 1, 1)
            denoised = self.denoise(x_next, sigma_cur, target_atoms_cur, h1_cur)
            d_cur = (x_next - denoised) / t_cur
            x_prime = (x_next + (t_next - t_cur) * d_cur) * comp_keep

            if t_next == 0:
                x_next = x_prime
            else:
                sigma_next = t_next.expand(bsz, 1, 1)
                denoised_next = self.denoise(x_prime, sigma_next, target_atoms_cur, h1_cur)
                d_prime = (x_prime - denoised_next) / t_next
                x_next = (x_next + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)) * comp_keep

            if return_intermediates:
                intermediates.append(x_next.clone())

        if return_intermediates:
            return x_next.squeeze(1), torch.stack(intermediates, dim=1)
        return x_next.squeeze(1)
