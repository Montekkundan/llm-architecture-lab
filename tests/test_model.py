import unittest
from dataclasses import replace

from architecture_lab import build_model, kv_cache_bytes, preset

try:
    import torch
    from architecture_lab.model import apply_rope
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None


@unittest.skipUnless(torch is not None, "Install the declared PyTorch dependency to run model checks")
class DecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)

    def test_rope_preserves_norm_and_relative_dot_products(self):
        x = torch.randn(2, 4, 5, 16, dtype=torch.float64)
        y = torch.randn_like(x)
        positions = torch.arange(5)
        rotated = apply_rope(x, positions, 10_000.0)
        torch.testing.assert_close(rotated.square().sum(-1), x.square().sum(-1))
        qk = rotated @ apply_rope(y, positions, 10_000.0).transpose(-2, -1)
        shifted = apply_rope(x, positions + 7, 10_000.0) @ apply_rope(y, positions + 7, 10_000.0).transpose(-2, -1)
        torch.testing.assert_close(qk, shifted)

    def test_all_attention_variants_match_chunked_cache(self):
        for name in ("pico-dense", "pico-gqa", "pico-mqa", "llama3-8b-tiny"):
            with self.subTest(name=name):
                spec = preset(name)
                model = build_model(spec).eval()
                ids = torch.randint(0, spec.vocab_size, (2, 7), dtype=torch.long)
                full = model(ids)
                pieces = []
                cache = None
                for start, stop in ((0, 3), (3, 5), (5, 7)):
                    logits, cache = model.forward_cached(ids[:, start:stop], cache)
                    pieces.append(logits)
                torch.testing.assert_close(full, torch.cat(pieces, dim=1), rtol=1e-5, atol=1e-5)
                cache.assert_prefix(ids)
                for layer in cache.layers:
                    self.assertEqual(tuple(layer.keys.shape), (2, spec.kv_heads, 7, spec.head_width))
                    self.assertEqual(tuple(layer.values.shape), (2, spec.kv_heads, 7, spec.head_width))

    def test_future_token_cannot_change_prefix(self):
        model = build_model(preset("pico-gqa")).eval()
        ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        changed = ids.clone()
        changed[:, 3:] = 8
        torch.testing.assert_close(model(ids)[:, :3], model(changed)[:, :3], rtol=1e-6, atol=1e-6)

    def test_backward_and_cache_invalidation(self):
        model = build_model(preset("pico-mqa"))
        ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        model(ids).sum().backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
        model.eval()
        _, cache = model.forward_cached(ids[:, :2])
        with torch.no_grad():
            model.token_embedding.weight[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "changed parameters"):
            model.forward_cached(ids[:, 2:], cache)

    def test_mechanism_presets_match_chunked_cache(self):
        from architecture_lab.attention import LatentCache
        from architecture_lab.position import YarnRoPE

        for name in ("pico-yarn", "pico-mla", "pico-mla-absorbed", "pico-mla-yarn-moe", "pico-moe"):
            with self.subTest(name=name):
                spec = preset(name)
                model = build_model(spec).eval()
                ids = torch.randint(0, spec.vocab_size, (2, 7), dtype=torch.long)
                full = model(ids)
                pieces = []
                cache = None
                for start, stop in ((0, 3), (3, 5), (5, 7)):
                    logits, cache = model.forward_cached(ids[:, start:stop], cache)
                    pieces.append(logits)
                torch.testing.assert_close(full, torch.cat(pieces, dim=1), rtol=1e-5, atol=1e-5)
                cache.assert_prefix(ids)
                if name == "pico-yarn":
                    self.assertIsInstance(model.blocks[0].attention.yarn, YarnRoPE)
                if spec.attention_mode == "mla":
                    self.assertTrue(all(isinstance(layer, LatentCache) for layer in cache.layers))
                    self.assertEqual(tuple(cache.layers[0].latent.shape), (2, 7, spec.mla_kv_rank))
                    payload = sum(layer.latent.numel() + layer.positional_keys.numel()
                                  for layer in cache.layers) * cache.layers[0].latent.element_size()
                    self.assertEqual(payload, kv_cache_bytes(spec, 2, 7, 4))
                if name == "pico-moe":
                    self.assertEqual(len(model.blocks[0].ffn.experts), spec.moe_experts)
                if name == "pico-mla-yarn-moe":
                    self.assertIsInstance(model.blocks[0].attention.yarn, YarnRoPE)
                    self.assertEqual(len(model.blocks[0].ffn.experts), spec.moe_experts)

    def test_combined_mla_moe_spec_has_finite_gradients_and_cache_guard(self):
        model = build_model(replace(preset("pico-mla-absorbed"), name="pico-mla-moe",
                                    ffn_mode="moe", moe_experts=4, moe_top_k=2))
        ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        logits = model(ids)
        logits.square().mean().backward()
        self.assertTrue(torch.isfinite(model.blocks[0].attention.kv_down.weight.grad).all())
        self.assertGreater(float(model.blocks[0].ffn.router.weight.grad.abs().sum()), 0)
        model.eval()
        _, cache = model.forward_cached(ids[:, :2])
        with torch.no_grad():
            model.token_embedding.weight[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "changed parameters"):
            model.forward_cached(ids[:, 2:], cache)

    def test_combined_mla_yarn_moe_is_causal(self):
        model = build_model(preset("pico-mla-yarn-moe")).eval()
        ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        changed = ids.clone()
        changed[:, 3:] = 8
        torch.testing.assert_close(model(ids)[:, :3], model(changed)[:, :3],
                                   rtol=1e-6, atol=1e-6)

    def test_yarn_cache_crosses_original_context(self):
        spec = preset("pico-yarn")
        model = build_model(spec).eval()
        ids = torch.randint(0, spec.vocab_size, (1, spec.yarn_original_context + 2))
        full = model(ids)
        first, cache = model.forward_cached(ids[:, :spec.yarn_original_context])
        last, cache = model.forward_cached(ids[:, spec.yarn_original_context:], cache)
        torch.testing.assert_close(full, torch.cat((first, last), dim=1), rtol=1e-5, atol=1e-5)
        self.assertEqual(cache.layers[0].keys.shape[-2], ids.shape[1])


if __name__ == "__main__":
    unittest.main()
