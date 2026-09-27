"""Executable correctness check for a teaching preset (requires PyTorch)."""

from __future__ import annotations

import argparse
import json

import torch

from . import PRESETS, build_model, kv_cache_bytes, preset
from .attention import LatentCache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="pico-gqa",
                        choices=tuple(name for name, spec in PRESETS.items() if spec.scale == "teaching"))
    args = parser.parse_args()
    spec = preset(args.preset)
    torch.manual_seed(7)
    model = build_model(spec).eval()
    ids = torch.randint(0, spec.vocab_size, (2, 7), dtype=torch.long)
    full = model(ids)
    pieces = []
    cache = None
    for start, stop in ((0, 3), (3, 5), (5, 7)):
        logits, cache = model.forward_cached(ids[:, start:stop], cache)
        pieces.append(logits)
    torch.testing.assert_close(full, torch.cat(pieces, dim=1), rtol=1e-5, atol=1e-5)
    assert cache is not None
    if spec.attention_mode == "mla":
        assert all(isinstance(layer, LatentCache) and
                   layer.latent.shape == (2, 7, spec.mla_kv_rank) and
                   layer.positional_keys.shape == (2, 1, 7, spec.mla_positional_width)
                   for layer in cache.layers)
    else:
        assert all(layer.keys.shape == (2, spec.kv_heads, 7, spec.head_width) for layer in cache.layers)
    cache.assert_prefix(ids)
    result = {
        "preset": spec.name,
        "schema_version": spec.schema_version,
        "attention": spec.attention_kind,
        "position": spec.position_mode,
        "ffn": spec.ffn_mode,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "theoretical_fp32_kv_payload_bytes": kv_cache_bytes(spec, 2, 7, 4),
        "cached_vs_full": "passed",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
