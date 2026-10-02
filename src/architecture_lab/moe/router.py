"""Sparse top-k dispatch with optional shared experts and explicit balancing references."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..model import SwiGLU


@dataclass(frozen=True)
class Routing:
    indices: torch.Tensor  # [..., top_k]
    weights: torch.Tensor  # [..., top_k]; sums to one per token unless gate="router_probability"
    counts: torch.Tensor  # [num_experts], counts assignments rather than unique tokens
    probabilities: torch.Tensor  # [..., experts], unbiased full-router softmax
    auxiliary_loss: torch.Tensor  # E * sum(selection fraction * mean probability)


class TopKMoE(nn.Module):
    """Token-choice top-k MoE.

    gate="selected_softmax" (default): w_k = softmax over the k selected logits, so the
    weights sum to one. For top_k=1 this is exactly 1 and its derivative p(1 - p) is
    exactly 0, so the task loss sends no gradient to the router; only the auxiliary
    balance loss (which reads the full-router probabilities) trains it.
    gate="router_probability": w_k = full-router softmax probability of the selected
    expert, not renormalised (Switch Transformer top-1 style), so the router receives a
    task gradient for every top_k. The weights then sum to less than one.
    """

    def __init__(self, width: int, ff_width: int, experts: int, top_k: int,
                 *, shared_experts: int = 0, gate: str = "selected_softmax") -> None:
        super().__init__()
        if min(width, ff_width, experts, top_k) <= 0 or top_k > experts:
            raise ValueError("Require positive dimensions and 1 <= top_k <= experts")
        if gate not in {"selected_softmax", "router_probability"}:
            raise ValueError("gate must be selected_softmax or router_probability")
        self.gate = gate
        self.router = nn.Linear(width, experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(width, ff_width) for _ in range(experts))
        if type(shared_experts) is not int or shared_experts < 0:
            raise ValueError("shared_experts must be a nonnegative integer")
        self.shared_experts = nn.ModuleList(SwiGLU(width, ff_width) for _ in range(shared_experts))
        self.register_buffer("selection_bias", torch.zeros(experts))
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Routing]:
        if x.ndim < 2 or x.shape[-1] != self.router.in_features:
            raise ValueError("Expected x[..., width]")
        tokens = x.reshape(-1, x.shape[-1])
        if tokens.shape[0] == 0:
            raise ValueError("At least one token is required")
        logits = self.router(tokens)
        indices = (logits + self.selection_bias).topk(self.top_k, dim=-1).indices
        probabilities = logits.softmax(dim=-1)
        if self.gate == "selected_softmax":
            weights = logits.gather(-1, indices).softmax(dim=-1)
        else:
            weights = probabilities.gather(-1, indices)
        result = torch.zeros_like(tokens)
        for expert_id, expert in enumerate(self.experts):
            token_idx, slot_idx = (indices == expert_id).nonzero(as_tuple=True)
            if token_idx.numel():
                contribution = expert(tokens[token_idx]) * weights[token_idx, slot_idx, None]
                result = result.index_add(0, token_idx, contribution)
        counts = torch.bincount(indices.reshape(-1), minlength=len(self.experts))
        for expert in self.shared_experts:
            result = result + expert(tokens)
        fractions = counts.to(probabilities.dtype) / (tokens.shape[0] * self.top_k)
        auxiliary_loss = len(self.experts) * (fractions * probabilities.mean(0)).sum()
        shape = x.shape[:-1]
        return result.reshape(x.shape), Routing(
            indices.reshape(*shape, self.top_k), weights.reshape(*shape, self.top_k), counts,
            probabilities.reshape(*shape, len(self.experts)), auxiliary_loss,
        )

    @torch.no_grad()
    def update_selection_bias(self, counts: torch.Tensor, rate: float) -> None:
        """A local loss-free selection heuristic; distributed training needs global counts."""
        if counts.shape != self.selection_bias.shape or (counts < 0).any() or not 0 <= rate < float("inf"):
            raise ValueError("Expected nonnegative per-expert counts and finite nonnegative rate")
        self.selection_bias.add_(rate * torch.sign(counts.float().mean() - counts.float()))
