import unittest

from architecture_lab import build_model, preset

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


if __name__ == "__main__":
    unittest.main()
