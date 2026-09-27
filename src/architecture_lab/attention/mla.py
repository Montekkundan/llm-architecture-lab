"""Small, unabsorbed MLA with a compressed KV cache and decoupled RoPE.

DeepSeek-V2 equations 9–19 motivate the layout. This version intentionally
reconstructs K/V from cached latents, favoring auditability over decode speed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from ..model import apply_rope


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
    ) -> None:
        super().__init__()
        if min(width, heads, content_width, positional_width, kv_rank, query_rank) <= 0:
            raise ValueError("All dimensions must be positive")
        if positional_width % 2 or rope_base <= 1 or not math.isfinite(rope_base):
            raise ValueError("RoPE width must be even and base finite > 1")
        self.heads = heads
        self.content_width = content_width
        self.positional_width = positional_width
        self.rope_base = rope_base
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
        q_position = apply_rope(q_position, positions, self.rope_base)

        latent = self.kv_down(x)
        positional_keys = self.k_position(x).unsqueeze(1)
        positional_keys = apply_rope(positional_keys, positions, self.rope_base)
        if past is not None:
            latent = torch.cat((past.latent, latent), dim=1)
            positional_keys = torch.cat((past.positional_keys, positional_keys), dim=2)
        total = latent.shape[1]
        k_content = self.k_content(latent).view(batch, total, self.heads, self.content_width).transpose(1, 2)
        v_content = self.v_content(latent).view(batch, total, self.heads, self.content_width).transpose(1, 2)
        content_scores = q_content @ k_content.transpose(-2, -1)
        position_scores = q_position @ positional_keys.transpose(-2, -1)
        scores = (content_scores + position_scores) / math.sqrt(self.content_width + self.positional_width)
        forbidden = torch.arange(total, device=x.device)[None, None, None, :] > positions[None, None, :, None]
        probabilities = scores.masked_fill(forbidden, float("-inf")).softmax(-1)
        attended = probabilities @ v_content
        result = self.output(attended.transpose(1, 2).contiguous().view(batch, length, -1))
        cache = LatentCache(latent, positional_keys) if use_cache else None
        return result, cache
