"""Versioned, auditable decoder specifications. This module has no torch dependency."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModelSpec:
    name: str
    scale: str
    vocab_size: int
    width: int
    layers: int
    heads: int
    kv_heads: int
    context: int
    ff_width: int
    rope_base: float = 10_000.0
    eps: float = 1e-5
    tie_embeddings: bool = True
    source_url: str = ""
    schema_version: int = SCHEMA_VERSION
    position_mode: str = "rope"
    yarn_original_context: int = 0
    yarn_scale: float = 1.0
    attention_mode: str = "standard"
    mla_content_width: int = 0
    mla_positional_width: int = 0
    mla_kv_rank: int = 0
    mla_query_rank: int = 0
    ffn_mode: str = "swiglu"
    moe_experts: int = 0
    moe_top_k: int = 0

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported ModelSpec schema_version: {self.schema_version}")
        if not self.name or self.name.strip() != self.name:
            raise ValueError("name must be nonempty and trimmed")
        if self.scale not in {"teaching", "published-reference"}:
            raise ValueError("scale must be teaching or published-reference")
        for field in ("vocab_size", "width", "layers", "heads", "kv_heads", "context", "ff_width"):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if (self.width // self.heads) % 2:
            raise ValueError("RoPE requires an even head width")
        if self.heads % self.kv_heads:
            raise ValueError("heads must be divisible by kv_heads")
        if not math.isfinite(self.rope_base) or self.rope_base <= 1:
            raise ValueError("rope_base must be finite and greater than one")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")
        if type(self.tie_embeddings) is not bool:
            raise ValueError("tie_embeddings must be boolean")
        if self.scale == "published-reference" and not self.source_url.startswith("https://"):
            raise ValueError("published references require an HTTPS primary source")
        if self.position_mode == "rope":
            if self.yarn_original_context != 0 or self.yarn_scale != 1.0:
                raise ValueError("Standard RoPE cannot carry YaRN settings")
        elif self.position_mode == "yarn":
            if (type(self.yarn_original_context) is not int or self.yarn_original_context <= 0
                    or type(self.yarn_scale) not in (int, float)
                    or not math.isfinite(self.yarn_scale) or self.yarn_scale <= 1
                    or not self.yarn_original_context < self.context <= self.yarn_original_context * self.yarn_scale):
                raise ValueError("YaRN needs an original context and a fixed scale covering context")
        else:
            raise ValueError("position_mode must be rope or yarn")
        mla_fields = (self.mla_content_width, self.mla_positional_width,
                      self.mla_kv_rank, self.mla_query_rank)
        if self.attention_mode == "standard":
            if any(mla_fields):
                raise ValueError("Standard attention cannot carry MLA dimensions")
        elif self.attention_mode == "mla":
            if (any(type(value) is not int or value <= 0 for value in mla_fields)
                    or self.mla_positional_width % 2 or self.kv_heads != self.heads
                    or self.position_mode != "rope"):
                raise ValueError("MLA needs positive ranks, even positional width, and decoupled RoPE")
        else:
            raise ValueError("attention_mode must be standard or mla")
        if self.ffn_mode == "swiglu":
            if self.moe_experts or self.moe_top_k:
                raise ValueError("Dense SwiGLU cannot carry MoE routing settings")
        elif self.ffn_mode == "moe":
            if (type(self.moe_experts) is not int or type(self.moe_top_k) is not int
                    or self.moe_experts < 2 or not 1 <= self.moe_top_k <= self.moe_experts):
                raise ValueError("MoE needs at least two experts and a valid top_k")
        else:
            raise ValueError("ffn_mode must be swiglu or moe")

    @property
    def attention_kind(self) -> str:
        if self.attention_mode == "mla":
            return "mla"
        if self.kv_heads == self.heads:
            return "mha"
        if self.kv_heads == 1:
            return "mqa"
        return "gqa"

    @property
    def head_width(self) -> int:
        return self.width // self.heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModelSpec:
        if not isinstance(value, dict):
            raise TypeError("ModelSpec must be a dictionary")
        return cls(**value)


_PICO = ModelSpec(
    name="pico-dense", scale="teaching", vocab_size=259, width=64,
    layers=2, heads=4, kv_heads=4, context=128, ff_width=176,
    source_url="https://arxiv.org/abs/2104.09864",
)

_LLAMA3_REFERENCE = ModelSpec(
    name="llama3-8b-reference", scale="published-reference", vocab_size=128_256,
    width=4096, layers=32, heads=32, kv_heads=8, context=8192,
    ff_width=14_336, rope_base=500_000.0, tie_embeddings=False,
    source_url="https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py",
)

PRESETS: dict[str, ModelSpec] = {
    _PICO.name: _PICO,
    "pico-gqa": replace(_PICO, name="pico-gqa", kv_heads=2,
                        source_url="https://arxiv.org/abs/2305.13245"),
    "pico-mqa": replace(_PICO, name="pico-mqa", kv_heads=1,
                        source_url="https://arxiv.org/abs/1911.02150"),
    "pico-yarn": replace(_PICO, name="pico-yarn", context=512,
                         position_mode="yarn", yarn_original_context=128, yarn_scale=4.0,
                         source_url="https://arxiv.org/abs/2309.00071"),
    "pico-mla": replace(_PICO, name="pico-mla", attention_mode="mla",
                        mla_content_width=16, mla_positional_width=8,
                        mla_kv_rank=16, mla_query_rank=32,
                        source_url="https://arxiv.org/abs/2405.04434"),
    "pico-moe": replace(_PICO, name="pico-moe", ffn_mode="moe",
                        moe_experts=4, moe_top_k=2,
                        source_url="https://arxiv.org/abs/2006.16668"),
    "llama3-8b-tiny": ModelSpec(
        name="llama3-8b-tiny", scale="teaching", vocab_size=259,
        width=128, layers=2, heads=8, kv_heads=2, context=128,
        ff_width=448, rope_base=500_000.0, tie_embeddings=False,
        source_url=_LLAMA3_REFERENCE.source_url,
    ),
    _LLAMA3_REFERENCE.name: _LLAMA3_REFERENCE,
}


def preset(name: str) -> ModelSpec:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown preset: {name}") from exc


def kv_cache_bytes(spec: ModelSpec, batch: int, tokens: int, bytes_per_element: int) -> int:
    """Theoretical K+V payload only; excludes allocator and tensor metadata."""
    for name, value in (("batch", batch), ("tokens", tokens), ("bytes_per_element", bytes_per_element)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if tokens > spec.context:
        raise ValueError("tokens exceed the configured context")
    if spec.attention_mode == "mla":
        return spec.layers * batch * tokens * (spec.mla_kv_rank + spec.mla_positional_width) * bytes_per_element
    return 2 * spec.layers * batch * tokens * spec.kv_heads * spec.head_width * bytes_per_element
