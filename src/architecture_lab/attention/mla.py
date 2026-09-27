"""Small MLA with a compressed KV cache and decoupled RoPE.

DeepSeek-V2 equations 9–19 motivate the layout. Cached inference can either
reconstruct K/V or absorb their up-projections into Q and the output weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from ..model import apply_rope
from ..position import YarnRoPE


@dataclass(frozen=True)
class LatentCache:
    latent: torch.Tensor  # [B, S, C], before K/V up-projections
    positional_keys: torch.Tensor  # [B, 1, S, Dr], after RoPE

    @property
    def length(self) -> int:
        return self.latent.shape[1]


class LatentAttention(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        content_width: int,
        positional_width: int,
        kv_rank: int,
        query_rank: int,
        *,
        rope_base: float = 10_000.0,
        inference_mode: str = "reconstruct",
        yarn_original_context: int = 0,
        yarn_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if min(width, heads, content_width, positional_width, kv_rank, query_rank) <= 0:
            raise ValueError("All dimensions must be positive")
        if positional_width % 2 or rope_base <= 1 or not math.isfinite(rope_base):
            raise ValueError("RoPE width must be even and base finite > 1")
        if inference_mode not in {"reconstruct", "absorbed"}:
            raise ValueError("inference_mode must be reconstruct or absorbed")
        if (yarn_original_context == 0) != (yarn_scale == 1.0):
            raise ValueError("YaRN requires both an original context and a scale > 1")
        self.heads = heads
        self.content_width = content_width
        self.positional_width = positional_width
        self.rope_base = rope_base
        self.inference_mode = inference_mode
        self.yarn = (YarnRoPE(positional_width, yarn_original_context, yarn_scale,
                              base=rope_base) if yarn_original_context else None)
        self.q_down = nn.Linear(width, query_rank, bias=False)
        self.q_content = nn.Linear(query_rank, heads * content_width, bias=False)
        self.q_position = nn.Linear(query_rank, heads * positional_width, bias=False)
        self.kv_down = nn.Linear(width, kv_rank, bias=False)
        self.k_content = nn.Linear(kv_rank, heads * content_width, bias=False)
        self.v_content = nn.Linear(kv_rank, heads * content_width, bias=False)
        self.k_position = nn.Linear(width, positional_width, bias=False)
        self.output = nn.Linear(heads * content_width, width, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        past: LatentCache | None = None,
        *,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, LatentCache | None]:
        if x.ndim != 3 or x.shape[1] == 0 or x.shape[2] != self.q_down.in_features:
            raise ValueError("Expected nonempty x[B,T,width]")
        batch, length, _ = x.shape
        offset = 0 if past is None else past.length
        if past is not None and (
            past.latent.shape != (batch, offset, self.kv_down.out_features)
            or past.positional_keys.shape != (batch, 1, offset, self.positional_width)
            or past.latent.device != x.device
            or past.latent.dtype != x.dtype
        ):
            raise ValueError("Past cache shape, dtype or device differs")
        positions = torch.arange(offset, offset + length, device=x.device)
        query_latent = self.q_down(x)
        q_content = self.q_content(query_latent).view(batch, length, self.heads, self.content_width).transpose(1, 2)
        q_position = self.q_position(query_latent).view(batch, length, self.heads, self.positional_width).transpose(1, 2)
        if self.yarn is None:
            q_position = apply_rope(q_position, positions, self.rope_base)
        else:
            q_position = self.yarn(q_position, positions)

        latent = self.kv_down(x)
        positional_keys = self.k_position(x).unsqueeze(1)
        if self.yarn is None:
            positional_keys = apply_rope(positional_keys, positions, self.rope_base)
        else:
            positional_keys = self.yarn(positional_keys, positions)
        if past is not None:
            latent = torch.cat((past.latent, latent), dim=1)
            positional_keys = torch.cat((past.positional_keys, positional_keys), dim=2)
        total = latent.shape[1]
        absorbed = use_cache and self.inference_mode == "absorbed" and not self.training
        if absorbed:
            key_weight = self.k_content.weight.view(self.heads, self.content_width, -1)
            query = torch.einsum("bhtd,hdc->bhtc", q_content, key_weight)
            content_scores = query @ latent.unsqueeze(1).transpose(-2, -1)
        else:
            k_content = self.k_content(latent).view(batch, total, self.heads, self.content_width).transpose(1, 2)
            content_scores = q_content @ k_content.transpose(-2, -1)
        position_scores = q_position @ positional_keys.transpose(-2, -1)
        scores = (content_scores + position_scores) / math.sqrt(self.content_width + self.positional_width)
        forbidden = torch.arange(total, device=x.device)[None, None, None, :] > positions[None, None, :, None]
        probabilities = scores.masked_fill(forbidden, float("-inf")).softmax(-1)
        if absorbed:
            attended_latent = probabilities @ latent.unsqueeze(1)
            value_weight = self.v_content.weight.view(self.heads, self.content_width, -1)
            output_weight = self.output.weight.view(-1, self.heads, self.content_width).permute(1, 0, 2)
            result = torch.einsum("bhtc,hwc->btw", attended_latent,
                                  torch.bmm(output_weight, value_weight))
        else:
            v_content = self.v_content(latent).view(batch, total, self.heads, self.content_width).transpose(1, 2)
            attended = probabilities @ v_content
            result = self.output(attended.transpose(1, 2).contiguous().view(batch, length, -1))
        cache = LatentCache(latent, positional_keys) if use_cache else None
        return result, cache
