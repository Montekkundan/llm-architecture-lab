"""Explicit teaching decoder: RoPE, RMSNorm, SwiGLU, MHA/MQA/GQA and KV cache.

This is a correctness reference, not a FlashAttention kernel or a published
model reproduction. Every preset shares the same trainer-facing logits API.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from .spec import ModelSpec

if TYPE_CHECKING:
    from .attention.mla import LatentCache


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    """Rotate adjacent pairs of x[B,H,T,D] at absolute positions[T]."""
    if x.ndim != 4 or positions.ndim != 1 or x.shape[-2] != positions.numel() or x.shape[-1] % 2:
        raise ValueError("Expected x[B,H,T,even D] and positions[T]")
    width = x.shape[-1]
    work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    inverse_frequencies = base ** (-torch.arange(0, width, 2, device=x.device, dtype=work_dtype) / width)
    angles = positions.to(device=x.device, dtype=work_dtype)[:, None] * inverse_frequencies[None, :]
    cosine = angles.cos()[None, None, :, :]
    sine = angles.sin()[None, None, :, :]
    even = x[..., 0::2].to(work_dtype)
    odd = x[..., 1::2].to(work_dtype)
    rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1)
    return rotated.flatten(-2).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().square().mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps).to(x.dtype)) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, width: int, ff_width: int):
        super().__init__()
        self.gate = nn.Linear(width, ff_width, bias=False)
        self.value = nn.Linear(width, ff_width, bias=False)
        self.output = nn.Linear(ff_width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(F.silu(self.gate(x)) * self.value(x))


@dataclass(frozen=True)
class LayerCache:
    # Keys have already received RoPE; values have not.
    keys: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class ModelCache:
    layers: tuple[LayerCache | LatentCache, ...]
    tokens: torch.Tensor
    owner: int
    parameter_versions: tuple[int, ...]

    def assert_prefix(self, tokens: torch.Tensor) -> None:
        if not torch.equal(self.tokens, tokens):
            raise ValueError("Cached token prefix changed; start with cache=None")


class CausalAttention(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.heads = spec.heads
        self.kv_heads = spec.kv_heads
        self.head_width = spec.head_width
        self.rope_base = spec.rope_base
        self.yarn = None
        if spec.position_mode == "yarn":
            from .position import YarnRoPE
            self.yarn = YarnRoPE(spec.head_width, spec.yarn_original_context,
                                 spec.yarn_scale, base=spec.rope_base)
        self.q_proj = nn.Linear(spec.width, spec.width, bias=False)
        self.k_proj = nn.Linear(spec.width, spec.kv_heads * spec.head_width, bias=False)
        self.v_proj = nn.Linear(spec.width, spec.kv_heads * spec.head_width, bias=False)
        self.o_proj = nn.Linear(spec.width, spec.width, bias=False)

    @staticmethod
    def _split(x: torch.Tensor, heads: int, head_width: int) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.reshape(batch, length, heads, head_width).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        past: LayerCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, LayerCache | None]:
        batch, length, width = x.shape
        q = self._split(self.q_proj(x), self.heads, self.head_width)
        k = self._split(self.k_proj(x), self.kv_heads, self.head_width)
        v = self._split(self.v_proj(x), self.kv_heads, self.head_width)
        if self.yarn is None:
            q = apply_rope(q, positions, self.rope_base)
            k = apply_rope(k, positions, self.rope_base)
        else:
            q = self.yarn(q, positions)
            k = self.yarn(k, positions)
        if past is not None:
            k = torch.cat((past.keys, k), dim=-2)
            v = torch.cat((past.values, v), dim=-2)

        # Repetition happens only for dot products; the stored cache stays at
        # kv_heads, which is the actual MQA/GQA memory saving.
        groups = self.heads // self.kv_heads
        expanded_k = k.repeat_interleave(groups, dim=1)
        expanded_v = v.repeat_interleave(groups, dim=1)
        scores = q @ expanded_k.transpose(-2, -1) / math.sqrt(self.head_width)
        keys = torch.arange(k.shape[-2], device=x.device)
        forbidden = keys[None, None, None, :] > positions[None, None, :, None]
        probabilities = scores.masked_fill(forbidden, float("-inf")).softmax(dim=-1)
        result = probabilities @ expanded_v
        result = result.transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.o_proj(result), LayerCache(k, v) if use_cache else None


class DecoderBlock(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.attention_mode = spec.attention_mode
        self.ffn_mode = spec.ffn_mode
        self.attention_norm = RMSNorm(spec.width, spec.eps)
        if spec.attention_mode == "mla":
            from .attention import LatentAttention
            self.attention = LatentAttention(
                spec.width, spec.heads, spec.mla_content_width,
                spec.mla_positional_width, spec.mla_kv_rank,
                spec.mla_query_rank, rope_base=spec.rope_base,
            )
        else:
            self.attention = CausalAttention(spec)
        self.ffn_norm = RMSNorm(spec.width, spec.eps)
        if spec.ffn_mode == "moe":
            from .moe import TopKMoE
            self.ffn = TopKMoE(spec.width, spec.ff_width, spec.moe_experts, spec.moe_top_k)
        else:
            self.ffn = SwiGLU(spec.width, spec.ff_width)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        past: LayerCache | LatentCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, LayerCache | LatentCache | None]:
        if self.attention_mode == "mla":
            attention_output, next_cache = self.attention(self.attention_norm(x), past, use_cache=use_cache)
        else:
            attention_output, next_cache = self.attention(self.attention_norm(x), positions, past, use_cache)
        x = x + attention_output
        ffn_output = self.ffn(self.ffn_norm(x))
        if self.ffn_mode == "moe":
            ffn_output, _ = ffn_output
        return x + ffn_output, next_cache


class DecoderLM(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        if spec.scale != "teaching":
            raise ValueError("Published-size references are metadata only")
        self.spec = spec
        self.token_embedding = nn.Embedding(spec.vocab_size, spec.width)
        self.blocks = nn.ModuleList(DecoderBlock(spec) for _ in range(spec.layers))
        self.final_norm = RMSNorm(spec.width, spec.eps)
        self.lm_head = nn.Linear(spec.width, spec.vocab_size, bias=False)
        if spec.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        for parameter in self.parameters():
            if parameter.ndim >= 2:
                nn.init.normal_(parameter, mean=0.0, std=0.02)

    def _run(
        self, input_ids: torch.Tensor, cache: ModelCache | None, use_cache: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, ModelCache]:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must be nonempty torch.long [batch, tokens]")
        if cache is not None and not use_cache:
            raise ValueError("A cache requires use_cache=True")
        versions = tuple(parameter._version for parameter in self.parameters())
        if cache is not None:
            if cache.owner != id(self) or cache.parameter_versions != versions:
                raise ValueError("Cache belongs to another model or changed parameters")
            if len(cache.layers) != len(self.blocks) or cache.tokens.shape[0] != input_ids.shape[0] or cache.tokens.device != input_ids.device:
                raise ValueError("Cache shape or device differs")
        offset = cache.tokens.shape[1] if cache is not None else 0
        length = input_ids.shape[1]
        if offset + length > self.spec.context:
            raise ValueError("Sequence exceeds configured context")

        positions = torch.arange(offset, offset + length, device=input_ids.device)
        x = self.token_embedding(input_ids)
        next_layers: list[LayerCache | LatentCache] = []
        for index, block in enumerate(self.blocks):
            previous = cache.layers[index] if cache is not None else None
            x, next_layer = block(x, positions, previous, use_cache)
            if next_layer is not None:
                next_layers.append(next_layer)
        logits = self.lm_head(self.final_norm(x))
        if not use_cache:
            return logits
        tokens = input_ids.clone() if cache is None else torch.cat((cache.tokens, input_ids), dim=1)
        return logits, ModelCache(tuple(next_layers), tokens, id(self), versions)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        result = self._run(input_ids, None, False)
        assert isinstance(result, torch.Tensor)
        return result

    @torch.inference_mode()
    def forward_cached(
        self, input_ids: torch.Tensor, cache: ModelCache | None = None,
    ) -> tuple[torch.Tensor, ModelCache]:
        if self.training:
            raise ValueError("Call model.eval() before cached inference")
        result = self._run(input_ids, cache, True)
        assert isinstance(result, tuple)
        return result
