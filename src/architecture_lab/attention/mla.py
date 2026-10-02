"""Small MLA with a compressed KV cache and decoupled RoPE.

DeepSeek-V2 equations 9–19 motivate the layout. Cached inference can either
reconstruct K/V or absorb their up-projections into Q and the output weights.

Two optional switches align the layer with the DeepSeek-V3 / Hugging Face layout and
are off by default so the original arithmetic and parameter count are unchanged:
latent_norm adds the q_a / kv_a RMSNorm on the compressed latents (the cache then
stores the normalised KV latent), and mscale_scope="logit" replaces the positional-only
YaRN scaling by softmax scale (c + r)^(-1/2) * m^2 over the whole logit with
m = 0.1 ln(s) + 1, rotating Q/K without the multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from ..model import RMSNorm, apply_rope
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
        latent_norm: bool = False,
        norm_eps: float = 1e-6,
        mscale_scope: str = "positional",
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
        if mscale_scope not in {"positional", "logit"}:
            raise ValueError("mscale_scope must be positional or logit")
        if mscale_scope == "logit" and not yarn_original_context:
            raise ValueError("mscale_scope='logit' requires YaRN")
        self.heads = heads
        self.content_width = content_width
        self.positional_width = positional_width
        self.rope_base = rope_base
        self.inference_mode = inference_mode
        self.mscale_scope = mscale_scope
        self.yarn = (YarnRoPE(positional_width, yarn_original_context, yarn_scale,
                              base=rope_base, apply_multiplier=mscale_scope == "positional")
                     if yarn_original_context else None)
        self.q_down = nn.Linear(width, query_rank, bias=False)
        self.q_content = nn.Linear(query_rank, heads * content_width, bias=False)
        self.q_position = nn.Linear(query_rank, heads * positional_width, bias=False)
        self.kv_down = nn.Linear(width, kv_rank, bias=False)
        self.k_content = nn.Linear(kv_rank, heads * content_width, bias=False)
        self.v_content = nn.Linear(kv_rank, heads * content_width, bias=False)
        self.k_position = nn.Linear(width, positional_width, bias=False)
        self.output = nn.Linear(heads * content_width, width, bias=False)
        self.q_norm = RMSNorm(query_rank, norm_eps) if latent_norm else None
        self.kv_norm = RMSNorm(kv_rank, norm_eps) if latent_norm else None

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
        if self.q_norm is not None:
            query_latent = self.q_norm(query_latent)
        q_content = self.q_content(query_latent).view(batch, length, self.heads, self.content_width).transpose(1, 2)
        q_position = self.q_position(query_latent).view(batch, length, self.heads, self.positional_width).transpose(1, 2)
        if self.yarn is None:
            q_position = apply_rope(q_position, positions, self.rope_base)
        else:
            q_position = self.yarn(q_position, positions)

        latent = self.kv_down(x)
        if self.kv_norm is not None:
            latent = self.kv_norm(latent)  # normalised before caching, as in DeepSeek-V3
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
        if self.mscale_scope == "logit":
            scores = scores * self.yarn.attention_multiplier ** 2  # whole-logit mscale^2
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
