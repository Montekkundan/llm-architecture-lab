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


if __name__ == "__main__":
    unittest.main()
