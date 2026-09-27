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

    @property
    def attention_kind(self) -> str:
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
    return 2 * spec.layers * batch * tokens * spec.kv_heads * spec.head_width * bytes_per_element
