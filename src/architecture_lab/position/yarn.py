"""Fixed-scale YaRN RoPE: NTK-by-parts frequencies plus attention scaling.

This is a small reference for Peng et al., arXiv:2309.00071, equations 17–22 (v2).
It deliberately omits dynamic scaling, whose cache requires re-rotating old keys.

Two ramps are available. ramp="dimension" (the default) is the formulation of the
authors' released code and of Hugging Face transformers (_compute_yarn_parameters):
with d(r) = D ln(L / (2 pi r)) / (2 ln b) the pair index that completes r turns over
the original context L, the extrapolation weight falls linearly in the pair index
from 1 at low = floor(d(beta)) to 0 at high = ceil(d(alpha)). ramp="rotations" is the
paper's printed equation, linear in the rotation count r_j = L theta_j / (2 pi).
Both give theta'_j = (1 - gamma_j) theta_j / s + gamma_j theta_j with gamma the
extrapolation weight; they differ only for the pairs between the two bounds.
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
        ramp: str = "dimension",
        apply_multiplier: bool = True,
    ) -> None:
        if head_width <= 0 or head_width % 2:
            raise ValueError("head_width must be positive and even")
        if original_context <= 0 or scale < 1 or not math.isfinite(scale):
            raise ValueError("original_context must be positive and finite scale >= 1")
        if base <= 1 or not math.isfinite(base) or not (0 <= alpha < beta):
            raise ValueError("Require finite base > 1 and 0 <= alpha < beta")
        if ramp not in {"dimension", "rotations"}:
            raise ValueError("ramp must be dimension or rotations")
        self.head_width = head_width
        self.original_context = original_context
        self.scale = scale
        self.base = base
        self.alpha = alpha
        self.beta = beta
        self.ramp = ramp
        # False leaves the 0.1 ln(s) + 1 factor to the caller, as DeepSeek-V3 MLA does
        # when it multiplies the whole logit by mscale^2 instead of rotating scaled Q/K.
        self.apply_multiplier = apply_multiplier

    def _correction_dim(self, rotations: float) -> float:
        """Pair index whose frequency completes `rotations` turns over original_context."""
        if rotations == 0:
            return math.inf
        return self.head_width * math.log(self.original_context / (rotations * 2 * math.pi)) / (2 * math.log(self.base))

    def correction_range(self) -> tuple[float, float]:
        """(low, high) pair indices of the dimension ramp; same clamping as the reference code.

        The reference clamps high to head_width - 1 (not head_width / 2 - 1, the last
        pair index) and the clamp is kept so the ramp matches it exactly.
        """
        low = max(math.floor(self._correction_dim(self.beta)), 0)
        high = math.ceil(min(self._correction_dim(self.alpha), self.head_width - 1))
        return low, high

    def frequencies(self, *, device: torch.device | None = None, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """One angular frequency per adjacent coordinate pair, high to low.

        theta'_j = theta_j * (g_j + (1 - g_j) / s) with g_j the extrapolation weight.
        """
        pair = torch.arange(self.head_width // 2, device=device, dtype=dtype)
        original = self.base ** (-2 * pair / self.head_width)
        if self.ramp == "rotations":
            rotations_in_original_context = self.original_context * original / (2 * math.pi)
            ramp = ((rotations_in_original_context - self.alpha) / (self.beta - self.alpha)).clamp(0, 1)
            return original * ((1 - ramp) / self.scale + ramp)
        low, high = self.correction_range()
        if low == high:
            high += 0.001  # the reference's guard against a zero-width ramp
        interpolation_weight = ((pair - low) / (high - low)).clamp(0, 1)
        return original * ((1 - interpolation_weight) + interpolation_weight / self.scale)

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
        if self.apply_multiplier:
            rotated = rotated * self.attention_multiplier
        return rotated.flatten(-2).to(x.dtype)
