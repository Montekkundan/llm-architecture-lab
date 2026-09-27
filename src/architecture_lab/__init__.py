"""Small decoder architecture lab for lessons 31–50."""

from .spec import ModelSpec, PRESETS, kv_cache_bytes, preset


def build_model(spec: ModelSpec):
    """Load torch only when a runnable model is requested."""
    if spec.scale != "teaching":
        raise ValueError("Published-size references are metadata only; choose a teaching preset")
    from .model import DecoderLM
    return DecoderLM(spec)


__all__ = ["ModelSpec", "PRESETS", "preset", "kv_cache_bytes", "build_model"]
