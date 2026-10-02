import unittest
from dataclasses import replace

from architecture_lab import ModelSpec, build_model, kv_cache_bytes, preset


class ModelSpecTests(unittest.TestCase):
    def test_presets_and_versioned_roundtrip(self):
        self.assertEqual(preset("pico-dense").attention_kind, "mha")
        self.assertEqual(preset("pico-gqa").attention_kind, "gqa")
        self.assertEqual(preset("pico-mqa").attention_kind, "mqa")
        for name in ("pico-dense", "pico-gqa", "pico-mqa", "llama3-8b-tiny", "llama3-8b-reference"):
            original = preset(name)
            self.assertEqual(ModelSpec.from_dict(original.to_dict()), original)
            self.assertEqual(original.schema_version, 1)

    def test_invalid_head_layout_and_version_are_rejected(self):
        baseline = preset("pico-gqa")
        for update in ({"width": 65}, {"kv_heads": 3}, {"width": 60}, {"schema_version": 2}, {"rope_base": 1.0}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                replace(baseline, **update)

    def test_published_size_reference_cannot_allocate(self):
        reference = preset("llama3-8b-reference")
        self.assertEqual(reference.scale, "published-reference")
        self.assertEqual(reference.attention_kind, "gqa")
        self.assertEqual(reference.heads // reference.kv_heads, 4)
        with self.assertRaisesRegex(ValueError, "metadata only"):
            build_model(reference)

    def test_cache_payload_tracks_stored_kv_heads(self):
        dense = preset("pico-dense")
        gqa = preset("pico-gqa")
        mqa = preset("pico-mqa")
        self.assertEqual(kv_cache_bytes(dense, 2, 7, 4), 2 * 2 * 2 * 7 * 4 * 16 * 4)
        self.assertEqual(kv_cache_bytes(dense, 2, 7, 4), 2 * kv_cache_bytes(gqa, 2, 7, 4))
        self.assertEqual(kv_cache_bytes(dense, 2, 7, 4), 4 * kv_cache_bytes(mqa, 2, 7, 4))

    def test_mechanism_presets_roundtrip_and_cache_accounting(self):
        for name, kind in (("pico-yarn", "mha"), ("pico-mla", "mla"),
                           ("pico-mla-absorbed", "mla"), ("pico-mla-yarn-moe", "mla"),
                           ("pico-moe", "mha")):
            with self.subTest(name=name):
                spec = preset(name)
                self.assertEqual(spec.attention_kind, kind)
                self.assertEqual(ModelSpec.from_dict(spec.to_dict()), spec)
        mla = preset("pico-mla")
        self.assertEqual(
            kv_cache_bytes(mla, 2, 7, 4),
            mla.layers * 2 * 7 * (mla.mla_kv_rank + mla.mla_positional_width) * 4,
        )
        original = preset("pico-dense")
        old_fields = {key: value for key, value in original.to_dict().items()
                      if key not in {"position_mode", "yarn_original_context", "yarn_scale",
                                     "attention_mode", "mla_content_width", "mla_positional_width",
                                     "mla_kv_rank", "mla_query_rank", "mla_inference_mode", "ffn_mode",
                                     "moe_experts", "moe_top_k", "mla_latent_norm",
                                     "mla_mscale_scope", "moe_gate"}}
        self.assertEqual(ModelSpec.from_dict(old_fields), original)

    def test_reference_aligned_options_roundtrip_and_need_their_mechanism(self):
        logit = replace(preset("pico-mla-yarn-moe"), mla_latent_norm=True, mla_mscale_scope="logit",
                        moe_gate="router_probability")
        self.assertEqual(ModelSpec.from_dict(logit.to_dict()), logit)
        for update in ({"mla_mscale_scope": "all"}, {"mla_latent_norm": 1}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                replace(preset("pico-mla"), **update)
        with self.assertRaises(ValueError):  # whole-logit mscale belongs to YaRN
            replace(preset("pico-mla"), mla_mscale_scope="logit")

    def test_invalid_mechanism_combinations_are_rejected(self):
        baseline = preset("pico-dense")
        invalid = (
            {"position_mode": "yarn"},
            {"position_mode": "yarn", "yarn_original_context": 128, "yarn_scale": 2.0},
            {"attention_mode": "mla"},
            {"ffn_mode": "moe"},
            {"ffn_mode": "moe", "moe_experts": 2, "moe_top_k": 3},
            {"mla_kv_rank": 4},
            {"mla_inference_mode": "absorbed"},
            {"position_mode": "unknown"},
            {"attention_mode": "unknown"},
            {"ffn_mode": "unknown"},
            {"position_mode": "yarn", "yarn_original_context": 128,
             "yarn_scale": "4", "context": 512},
            {"ffn_mode": "moe", "moe_experts": 4, "moe_top_k": True},
            {"attention_mode": "mla", "mla_content_width": 16,
             "mla_positional_width": 8, "mla_kv_rank": 16, "mla_query_rank": 32,
             "mla_inference_mode": "unknown"},
            {"mla_latent_norm": True},
            {"mla_mscale_scope": "logit"},
            {"moe_gate": "router_probability"},
            {"ffn_mode": "moe", "moe_experts": 4, "moe_top_k": 1, "moe_gate": "hard"},
        )
        for update in invalid:
            with self.subTest(update=update), self.assertRaises(ValueError):
                replace(baseline, **update)


if __name__ == "__main__":
    unittest.main()
