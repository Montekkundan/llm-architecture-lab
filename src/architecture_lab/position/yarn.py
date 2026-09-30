"""Fixed-scale YaRN RoPE: NTK-by-parts frequencies plus attention scaling.

This is a small reference for Peng et al., arXiv:2309.00071, equations 17–22 (v2).
It deliberately omits dynamic scaling, whose cache requires re-rotating old keys.
"""

from __future__ import annotations

import math

import torch


class YarnRoPE:
    def __init__(
        self,
        head_width: int,
        original_context: int,
        scale: float,
        *,
        base: float = 10_000.0,
        alpha: float = 1.0,
        beta: float = 32.0,
    ) -> None:
        if head_width <= 0 or head_width % 2:
            raise ValueError("head_width must be positive and even")
        if original_context <= 0 or scale < 1 or not math.isfinite(scale):
            raise ValueError("original_context must be positive and finite scale >= 1")
        if base <= 1 or not math.isfinite(base) or not (0 <= alpha < beta):
            raise ValueError("Require finite base > 1 and 0 <= alpha < beta")
        self.head_width = head_width
        self.original_context = original_context
        self.scale = scale
        self.base = base
        self.alpha = alpha
        self.beta = beta

    def frequencies(self, *, device: torch.device | None = None, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """One angular frequency per adjacent coordinate pair, high to low."""
        pair = torch.arange(self.head_width // 2, device=device, dtype=dtype)
        original = self.base ** (-2 * pair / self.head_width)
        rotations_in_original_context = self.original_context * original / (2 * math.pi)
        ramp = ((rotations_in_original_context - self.alpha) / (self.beta - self.alpha)).clamp(0, 1)
        return original * ((1 - ramp) / self.scale + ramp)

    @property
    def attention_multiplier(self) -> float:
        """Multiplier on Q *and* K; the dot product gets its square."""
        return 1.0 + 0.1 * math.log(self.scale)

    def __call__(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate adjacent pairs in x[B,H,T,D] with a fixed extension scale."""
        if x.ndim != 4 or x.shape[-1] != self.head_width or positions.ndim != 1 or positions.numel() != x.shape[-2]:
            raise ValueError("Expected x[B,H,T,head_width] and positions[T]")
        work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        theta = self.frequencies(device=x.device, dtype=work_dtype)
        phase = positions.to(device=x.device, dtype=work_dtype)[:, None] * theta[None, :]
        cosine, sine = phase.cos()[None, None], phase.sin()[None, None]
        even, odd = x[..., 0::2].to(work_dtype), x[..., 1::2].to(work_dtype)
        rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1)
        return (rotated.flatten(-2) * self.attention_multiplier).to(x.dtype)
