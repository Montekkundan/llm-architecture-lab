import math
import unittest

import torch

from architecture_lab.attention import LatentAttention
from architecture_lab.model import apply_rope
from architecture_lab.moe import TopKMoE
from architecture_lab.position import YarnRoPE


class MechanismTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(1)

    def test_yarn_fixed_scale_and_relative_geometry(self):
        x = torch.randn(2, 3, 5, 16, dtype=torch.float64)
        y = torch.randn_like(x)
        positions = torch.arange(5)
        unchanged = YarnRoPE(16, 4096, 1.0)
        torch.testing.assert_close(unchanged(x, positions), apply_rope(x, positions, 10_000.0))

        yarn = YarnRoPE(16, 4096, 4.0)
        frequencies = yarn.frequencies()
        original = unchanged.frequencies()
        self.assertAlmostEqual(float(frequencies[0]), float(original[0]))
        self.assertAlmostEqual(float(frequencies[-1]), float(original[-1] / 4))
        self.assertTrue(torch.all(frequencies[1:] <= frequencies[:-1]))
        factor = yarn.attention_multiplier
        torch.testing.assert_close(yarn(x, positions).square().sum(-1), x.square().sum(-1) * factor**2)
        scores = yarn(x, positions) @ yarn(y, positions).transpose(-2, -1)
        shifted = yarn(x, positions + 11) @ yarn(y, positions + 11).transpose(-2, -1)
        torch.testing.assert_close(scores, shifted)
        with self.assertRaises(ValueError):
            YarnRoPE(15, 4096, 4.0)

    def test_mla_cache_matches_full_and_all_parameters_train(self):
        model = LatentAttention(16, 4, 4, 4, kv_rank=6, query_rank=8).double()
        x = torch.randn(2, 7, 16, dtype=torch.float64, requires_grad=True)
        full, absent = model(x)
        self.assertIsNone(absent)
        full.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

        with torch.no_grad():
            first, cache = model(x[:, :3], use_cache=True)
            self.assertEqual(tuple(cache.latent.shape), (2, 3, 6))
            self.assertEqual(tuple(cache.positional_keys.shape), (2, 1, 3, 4))
            self.assertLess(6 + 4, 2 * 4 * 4)  # latent + positional key vs dense K/V scalars per token
            second, cache = model(x[:, 3:5], cache, use_cache=True)
            third, cache = model(x[:, 5:], cache, use_cache=True)
            self.assertEqual(cache.length, 7)
            torch.testing.assert_close(torch.cat((first, second, third), dim=1), full, rtol=1e-12, atol=1e-12)
            changed = x.detach().clone()
            changed[:, 4:] = torch.randn_like(changed[:, 4:])
            prefix, _ = model(changed)
            torch.testing.assert_close(prefix[:, :4], full[:, :4], rtol=1e-12, atol=1e-12)
            with self.assertRaises(ValueError):
                model(x[:1, 3:4], cache, use_cache=True)

    def test_moe_dispatch_conserves_weight_and_has_gradients(self):
        moe = TopKMoE(width=8, ff_width=12, experts=4, top_k=2).double()
        x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
        output, routing = moe(x)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(tuple(routing.indices.shape), (2, 5, 2))
        self.assertEqual(int(routing.counts.sum()), 20)
        torch.testing.assert_close(routing.weights.sum(-1), torch.ones(2, 5, dtype=torch.float64))
        manual = torch.zeros_like(output)
        for b in range(2):
            for t in range(5):
                for slot in range(2):
                    expert_id = int(routing.indices[b, t, slot])
                    manual[b, t] += routing.weights[b, t, slot] * moe.experts[expert_id](x[b, t])
        torch.testing.assert_close(output, manual)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(moe.router.weight.grad.abs().sum()), 0)
        for expert_id, count in enumerate(routing.counts.tolist()):
            if count:
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in moe.experts[expert_id].parameters()))


if __name__ == "__main__":
    unittest.main()
