"""Sparse top-k expert dispatch without capacity limits or auxiliary losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..model import SwiGLU


@dataclass(frozen=True)
class Routing:
    indices: torch.Tensor  # [..., top_k]
    weights: torch.Tensor  # [..., top_k], sums to one per token
    counts: torch.Tensor  # [num_experts], counts assignments rather than unique tokens


class TopKMoE(nn.Module):
    def __init__(self, width: int, ff_width: int, experts: int, top_k: int) -> None:
        super().__init__()
        if min(width, ff_width, experts, top_k) <= 0 or top_k > experts:
            raise ValueError("Require positive dimensions and 1 <= top_k <= experts")
        self.router = nn.Linear(width, experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(width, ff_width) for _ in range(experts))
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Routing]:
        if x.ndim < 2 or x.shape[-1] != self.router.in_features:
            raise ValueError("Expected x[..., width]")
        tokens = x.reshape(-1, x.shape[-1])
        if tokens.shape[0] == 0:
            raise ValueError("At least one token is required")
        chosen_logits, indices = self.router(tokens).topk(self.top_k, dim=-1)
        weights = chosen_logits.softmax(dim=-1)
        result = torch.zeros_like(tokens)
        for expert_id, expert in enumerate(self.experts):
            token_idx, slot_idx = (indices == expert_id).nonzero(as_tuple=True)
            if token_idx.numel():
                contribution = expert(tokens[token_idx]) * weights[token_idx, slot_idx, None]
                result = result.index_add(0, token_idx, contribution)
        counts = torch.bincount(indices.reshape(-1), minlength=len(self.experts))
        shape = x.shape[:-1]
        return result.reshape(x.shape), Routing(indices.reshape(*shape, self.top_k), weights.reshape(*shape, self.top_k), counts)
