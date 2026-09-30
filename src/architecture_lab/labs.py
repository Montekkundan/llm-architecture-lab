"""Differentiable CPU references for lessons 38–50; no optimized GPU kernels."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence
import math

import torch
from torch import nn
from torch.nn import functional as F

from .spec import ModelSpec


def causal_window_mask(positions: torch.Tensor, total_keys: int,
                       window: int | None = None) -> torch.Tensor:
    """Allowed keys for absolute query positions; window includes the current token."""
    if positions.ndim != 1 or positions.numel() == 0 or positions.dtype != torch.long:
        raise ValueError("positions must be nonempty int64 [queries]")
    if type(total_keys) is not int or total_keys < 1 or (positions < 0).any() or (positions >= total_keys).any():
        raise ValueError("Query positions must index the key prefix")
    if window is not None and (type(window) is not int or window < 1):
        raise ValueError("window must be a positive integer")
    distance = positions[:, None] - torch.arange(total_keys, device=positions.device)[None, :]
    return distance >= 0 if window is None else (distance >= 0) & (distance < window)


def selected_block_mask(block_scores: torch.Tensor, block_width: int,
                        blocks_to_keep: int, positions: torch.Tensor) -> torch.Tensor:
    """Include the current block in the budget, then rank legal blocks; ties prefer lower IDs.

    The caller must compute each score from information available at that query.
    A causal output mask cannot repair a selector that inspected future content.
    """
    if block_scores.ndim != 2 or block_scores.shape[0] != positions.numel() or not torch.isfinite(block_scores).all():
        raise ValueError("Expected finite scores[queries, blocks]")
    if type(block_width) is not int or type(blocks_to_keep) is not int or min(block_width, blocks_to_keep) < 1:
        raise ValueError("block_width and blocks_to_keep must be positive integers")
    total_keys = block_scores.shape[1] * block_width
    causal = causal_window_mask(positions, total_keys)
    selected = torch.zeros_like(causal)
    key_blocks = torch.arange(total_keys, device=positions.device) // block_width
    for row, position in enumerate(positions.tolist()):
        current = position // block_width
        ranked = torch.argsort(block_scores[row, :current], descending=True, stable=True)
        chosen = torch.cat((ranked[:blocks_to_keep - 1], ranked.new_tensor([current])))
        selected[row] = torch.isin(key_blocks, chosen) & causal[row]
    return selected


def _attention_shapes(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      mask: torch.Tensor) -> None:
    if (q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or q.shape[:2] != k.shape[:2]
            or k.shape[:3] != v.shape[:3] or q.shape[-1] != k.shape[-1]
            or mask.dtype != torch.bool or mask.shape != (q.shape[-2], k.shape[-2])):
        raise ValueError("Expected Q[B,H,T,D], K[B,H,S,D], V[B,H,S,Dv], boolean mask[T,S]")
    if not mask.any(-1).all():
        raise ValueError("Every query needs at least one allowed key")


def masked_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """Dense-then-mask oracle; sparse masks alone do not reduce this computation."""
    _attention_shapes(q, k, v, mask)
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    scores = q.to(dtype) @ k.to(dtype).transpose(-2, -1) / math.sqrt(q.shape[-1])
    return (scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v.to(dtype)).to(q.dtype)


@dataclass(frozen=True)
class LinearState:
    matrix: torch.Tensor  # [B,H,Dk,Dv]
    normalizer: torch.Tensor  # [B,H,Dk]


def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     state: LinearState | None = None) -> tuple[torch.Tensor, LinearState]:
    """Normalized ELU+1 kernel attention, inclusive update-then-query recurrence."""
    if (q.ndim != 4 or q.shape != k.shape or q.shape[:3] != v.shape[:3]
            or q.shape[-2] == 0 or q.device != k.device or q.device != v.device):
        raise ValueError("Expected Q/K[B,H,T,Dk] and V[B,H,T,Dv] on one device")
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    queries, keys, values = F.elu(q.to(dtype)) + 1, F.elu(k.to(dtype)) + 1, v.to(dtype)
    matrix_shape = (*q.shape[:2], q.shape[-1], v.shape[-1])
    normalizer_shape = (*q.shape[:2], q.shape[-1])
    if state is None:
        matrix = q.new_zeros(matrix_shape, dtype=dtype)
        normalizer = q.new_zeros(normalizer_shape, dtype=dtype)
    else:
        if (state.matrix.shape != matrix_shape or state.normalizer.shape != normalizer_shape
                or state.matrix.device != q.device or state.normalizer.device != q.device
                or state.matrix.dtype != dtype or state.normalizer.dtype != dtype):
            raise ValueError("Recurrent state shape, dtype or device differs")
        matrix, normalizer = state.matrix, state.normalizer
    outputs = []
    for token in range(q.shape[-2]):
        kt, vt, qt = keys[:, :, token], values[:, :, token], queries[:, :, token]
        matrix = matrix + kt[..., :, None] * vt[..., None, :]
        normalizer = normalizer + kt
        numerator = torch.einsum('bhk,bhkv->bhv', qt, matrix)
        denominator = (qt * normalizer).sum(-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
        outputs.append(numerator / denominator)
    return torch.stack(outputs, dim=-2).to(q.dtype), LinearState(matrix, normalizer)


def qk_normalize(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMS normalization over head features only, preserving token independence."""
    if x.ndim != 4 or eps <= 0:
        raise ValueError("Expected x[B,H,T,D] and positive eps")
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    work = x.to(dtype)
    return (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


def residual_branch(x: torch.Tensor, branch: Callable[[torch.Tensor], torch.Tensor],
                    norm: Callable[[torch.Tensor], torch.Tensor], mode: str,
                    *, scale: float | torch.Tensor = 1.0,
                    gate: torch.Tensor | None = None) -> torch.Tensor:
    """Explicit residual equations; norm placement is part of the architecture."""
    if mode not in {'pre', 'after_add', 'branch_output', 'sandwich'}:
        raise ValueError("Unknown normalization placement")
    result = branch(norm(x) if mode in {'pre', 'sandwich'} else x)
    if mode in {'branch_output', 'sandwich'}:
        result = norm(result)
    if gate is not None:
        result = result * torch.sigmoid(gate)
    output = x + scale * result
    return norm(output) if mode == 'after_add' else output


class GeluFFN(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.input = nn.Linear(width, hidden, bias=False)
        self.output = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(F.gelu(self.input(x)))


class MultiTokenHeads(nn.Module):
    """Independent horizon heads; this is not DeepSeek's sequential MTP module."""
    def __init__(self, width: int, vocab_size: int, horizons: int = 2):
        super().__init__()
        if min(width, vocab_size, horizons) < 1:
            raise ValueError("Dimensions and horizons must be positive")
        self.heads = nn.ModuleList(nn.Linear(width, vocab_size, bias=False) for _ in range(horizons))

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(head(hidden) for head in self.heads)


def multitoken_loss(logits: Sequence[torch.Tensor], tokens: torch.Tensor,
                    document_ids: torch.Tensor, *, weights: Sequence[float] | None = None
                    ) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Horizon h predicts token t+h+1 without crossing any document boundary."""
    if not logits or tokens.ndim != 2 or document_ids.shape != tokens.shape:
        raise ValueError("Expected horizon logits and tokens/document_ids[B,T]")
    weights = tuple(1.0 for _ in logits) if weights is None else tuple(weights)
    if len(weights) != len(logits) or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("One finite nonnegative weight is required per horizon")
    total = logits[0].sum() * 0
    counts = []
    for horizon, (prediction, weight) in enumerate(zip(logits, weights), start=1):
        if prediction.ndim != 3 or prediction.shape[:2] != tokens.shape:
            raise ValueError("Each prediction must be logits[B,T,V]")
        length = max(0, tokens.shape[1] - horizon)
        valid = torch.ones((tokens.shape[0], length), device=tokens.device, dtype=torch.bool)
        for offset in range(1, horizon + 1):
            valid &= document_ids[:, :length] == document_ids[:, offset:offset + length]
        count = int(valid.sum())
        counts.append(count)
        if count:
            total = total + weight * F.cross_entropy(prediction[:, :length][valid].to(torch.float64 if prediction.dtype == torch.float64 else torch.float32),
                                                     tokens[:, horizon:][valid])
        else:
            total = total + prediction.sum() * 0
    return total, tuple(counts)


def online_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     mask: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Online softmax over key tiles; exact attention math with readable tensor ops.

    This reference still accepts a full boolean mask and its Python/autograd
    graph is not a fused FlashAttention kernel or a memory/latency benchmark.
    """
    _attention_shapes(q, k, v, mask)
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be positive")
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    query, key, value = q.to(dtype), k.to(dtype), v.to(dtype)
    maximum = query.new_full((*q.shape[:3], 1), -torch.inf)
    denominator = query.new_zeros((*q.shape[:3], 1))
    numerator = query.new_zeros((*q.shape[:3], v.shape[-1]))
    for start in range(0, k.shape[-2], block_size):
        stop = min(start + block_size, k.shape[-2])
        tile_mask = mask[:, start:stop]
        scores = query @ key[:, :, start:stop].transpose(-2, -1) / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(~tile_mask, -torch.inf)
        next_maximum = torch.maximum(maximum, scores.amax(-1, keepdim=True))
        safe_maximum = next_maximum.masked_fill(~torch.isfinite(next_maximum), 0)
        previous_scale = torch.exp(maximum - safe_maximum)
        tile_weights = torch.exp(scores - safe_maximum)
        numerator = numerator * previous_scale + tile_weights @ value[:, :, start:stop]
        denominator = denominator * previous_scale + tile_weights.sum(-1, keepdim=True)
        maximum = next_maximum
    return (numerator / denominator).to(q.dtype)


def parameter_count(spec: ModelSpec) -> int:
    """Exact trainable count for this repository's DecoderLM, not arbitrary families."""
    d, h, f = spec.width, spec.heads, spec.ff_width
    embeddings = spec.vocab_size * d * (1 if spec.tie_embeddings else 2)
    if spec.attention_mode == 'standard':
        attention = 2 * d * d + 2 * d * spec.kv_heads * spec.head_width
    else:
        c, r, p, a = spec.mla_content_width, spec.mla_positional_width, spec.mla_kv_rank, spec.mla_query_rank
        attention = d * a + a * h * (c + r) + d * p + 2 * p * h * c + d * r + h * c * d
    ffn = 3 * d * f
    if spec.ffn_mode == 'moe':
        ffn = spec.moe_experts * ffn + d * spec.moe_experts
    return embeddings + d + spec.layers * (2 * d + attention + ffn)


def training_matmul_flops(active_parameters: int, tokens: int) -> int:
    """6ND estimate: omits attention scores, nonmatmul ops, recompute and communication."""
    if type(active_parameters) is not int or type(tokens) is not int or min(active_parameters, tokens) < 1:
        raise ValueError("active_parameters and tokens must be positive integers")
    return 6 * active_parameters * tokens


def ring_allreduce_bytes(payload_bytes: int, ranks: int) -> float:
    """Ideal ring transfer per rank for reduce-scatter plus all-gather."""
    if type(payload_bytes) is not int or type(ranks) is not int or payload_bytes < 0 or ranks < 1:
        raise ValueError("Require nonnegative integer payload_bytes and positive ranks")
    return 2 * (ranks - 1) * payload_bytes / ranks


def lab_report() -> dict[str, object]:
    """Small seeded output checkpoints; timings and capability scores are not reported."""
    from . import build_model, kv_cache_bytes, preset
    from .model import SwiGLU
    from .moe import TopKMoE

    torch.manual_seed(38)
    torch.set_num_threads(1)
    q, k, v = [torch.randn(1, 2, 7, 4, dtype=torch.float64) for _ in range(3)]
    mask = causal_window_mask(torch.arange(7), 7, window=3)
    dense = masked_attention(q, k, v, mask)
    tiled = online_attention(q, k, v, mask, block_size=2)
    full, state = linear_attention(q, k, v)
    first, prefix = linear_attention(q[:, :, :3], k[:, :, :3], v[:, :, :3])
    last, _ = linear_attention(q[:, :, 3:], k[:, :, 3:], v[:, :, 3:], prefix)
    spec = preset("pico-gqa")
    model = build_model(spec).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        fp32 = model(ids)
        bf16 = model.to(torch.bfloat16)(ids).float()
    heads = MultiTokenHeads(8, 11, 2)
    _, counts = multitoken_loss(heads(torch.randn(1, 5, 8)),
                               torch.tensor([[1, 2, 3, 4, 5]]),
                               torch.tensor([[0, 0, 1, 1, 1]]))
    moe = TopKMoE(8, 12, 4, 2, shared_experts=1)
    _, routing = moe(torch.randn(2, 5, 8))
    block_mask = selected_block_mask(torch.ones(3, 4), 2, 2, torch.tensor([0, 2, 7]))
    return {
        "torch_version": torch.__version__, "device": "cpu",
        "local_allowed_keys_at_positions_3_4": causal_window_mask(torch.tensor([3, 4]), 5, 2).sum(-1).tolist(),
        "selected_keys_at_positions_0_2_7": [row.nonzero().flatten().tolist() for row in block_mask],
        "linear_state_scalars": state.matrix.numel() + state.normalizer.numel(),
        "linear_chunk_max_abs_error": float((full - torch.cat((first, last), -2)).abs().max()),
        "gelu_swiglu_matched_parameters": [sum(p.numel() for p in module.parameters()) for module in (GeluFFN(64, 192), SwiGLU(64, 128))],
        "expert_assignments": int(routing.counts.sum()),
        "maximum_gate_sum_error": float((routing.weights.sum(-1) - 1).abs().max().detach()),
        "multi_token_valid_counts": counts,
        "online_dense_max_abs_error": float((dense - tiled).abs().max()),
        "bf16_fp32_logits_max_abs_error": float((fp32 - bf16).abs().max()),
        "pico_gqa_parameters": parameter_count(spec),
        "pico_gqa_fp32_kv_bytes_B2_T7": kv_cache_bytes(spec, 2, 7, 4),
        "ideal_ring_bytes_8MiB_4ranks": ring_allreduce_bytes(8 * 1024 * 1024, 4),
    }
