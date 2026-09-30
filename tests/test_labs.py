import unittest

import torch
from torch.nn import functional as F

from architecture_lab import build_model, preset
from architecture_lab.labs import (
    GeluFFN, LinearState, MultiTokenHeads, causal_window_mask, linear_attention,
    masked_attention, multitoken_loss, online_attention, parameter_count,
    qk_normalize, residual_branch, ring_allreduce_bytes, selected_block_mask,
)
from architecture_lab.model import SwiGLU
from architecture_lab.moe import TopKMoE


class LabTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(38)
        torch.set_num_threads(1)

    def test_window_and_global_masks_use_absolute_positions(self):
        positions = torch.tensor([3, 4])
        local = causal_window_mask(positions, 5, window=2)
        self.assertEqual(local.tolist(), [[False, False, True, True, False],
                                        [False, False, False, True, True]])
        global_mask = causal_window_mask(positions, 5)
        self.assertEqual(global_mask.sum(-1).tolist(), [4, 5])
        with self.assertRaises(ValueError):
            causal_window_mask(positions, 5, window=0)

    def test_selected_blocks_keep_self_and_causality_with_stable_ties(self):
        scores = torch.ones(3, 4)
        mask = selected_block_mask(scores, block_width=2, blocks_to_keep=2,
                                   positions=torch.tensor([0, 2, 7]))
        self.assertEqual(mask[0].nonzero().flatten().tolist(), [0])
        self.assertEqual(mask[1].nonzero().flatten().tolist(), [0, 1, 2])
        self.assertEqual(mask[2].nonzero().flatten().tolist(), [0, 1, 6, 7])

    def test_sparse_attention_matches_dense_then_mask_and_is_causal(self):
        q, k, v = [torch.randn(1, 2, 7, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        mask = selected_block_mask(torch.randn(7, 4), 2, 2, torch.arange(7))[:, :7]
        actual = masked_attention(q, k, v, mask)
        expected = ((q @ k.transpose(-2, -1) / 2).masked_fill(~mask, -torch.inf).softmax(-1) @ v)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in (q, k, v)))
        changed_v = v.detach().clone()
        changed_v[:, :, 3:] += 50
        torch.testing.assert_close(actual[:, :, :3], masked_attention(q, k, changed_v, mask)[:, :, :3])

    def test_linear_scan_and_chunked_state_match_outputs_and_gradients(self):
        q, k, v = [torch.randn(2, 3, 7, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        expected, state = linear_attention(q, k, v)
        first, prefix = linear_attention(q[:, :, :3], k[:, :, :3], v[:, :, :3])
        last, chunk_state = linear_attention(q[:, :, 3:], k[:, :, 3:], v[:, :, 3:], prefix)
        actual = torch.cat((first, last), -2)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(chunk_state.matrix, state.matrix)
        torch.testing.assert_close(chunk_state.normalizer, state.normalizer)
        expected_grads = torch.autograd.grad(expected.sum(), (q, k, v), retain_graph=True)
        actual_grads = torch.autograd.grad(actual.sum(), (q, k, v))
        for left, right in zip(expected_grads, actual_grads):
            torch.testing.assert_close(left, right)
        self.assertEqual(tuple(state.matrix.shape), (2, 3, 4, 4))

    def test_linear_current_token_contributes_and_state_shape_is_checked(self):
        q, k, v = torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1, 2), torch.tensor([[[[2., 3.]]]])
        actual, _ = linear_attention(q, k, v)
        torch.testing.assert_close(actual, v)
        with self.assertRaises(ValueError):
            linear_attention(q, k, v, LinearState(torch.zeros(1, 1, 3, 2), torch.zeros(1, 1, 2)))

    def test_qk_norm_does_not_mix_tokens(self):
        x = torch.randn(2, 3, 5, 4)
        changed = x.clone()
        changed[:, :, 3:] *= 99
        torch.testing.assert_close(qk_normalize(x)[:, :, :3], qk_normalize(changed)[:, :, :3])
        self.assertTrue(torch.isfinite(qk_normalize(torch.zeros_like(x))).all())

    def test_norm_placements_have_different_zero_branch_outputs(self):
        x = torch.tensor([2., 4.], requires_grad=True)
        norm = lambda z: z * 2
        zero = lambda z: z * 0
        torch.testing.assert_close(residual_branch(x, zero, norm, 'pre'), x)
        torch.testing.assert_close(residual_branch(x, zero, norm, 'branch_output'), x)
        torch.testing.assert_close(residual_branch(x, zero, norm, 'after_add'), 2 * x)
        y = residual_branch(x, lambda z: z.square(), norm, 'sandwich')
        torch.testing.assert_close(y, x + 8 * x.square())

    def test_zero_residual_scale_blocks_branch_but_not_scale_gradient(self):
        x = torch.tensor([1., 2.], requires_grad=True)
        weight = torch.tensor(3., requires_grad=True)
        scale = torch.tensor(0., requires_grad=True)
        residual_branch(x, lambda z: weight * z, lambda z: z, 'pre', scale=scale).sum().backward()
        self.assertEqual(float(weight.grad), 0.)
        self.assertEqual(float(scale.grad), 9.)
        self.assertEqual(x.grad.tolist(), [1., 1.])

    def test_gelu_and_swiglu_parameter_budgets(self):
        gelu = GeluFFN(64, 192)
        swiglu = SwiGLU(64, 128)
        self.assertEqual(sum(p.numel() for p in gelu.parameters()), 24576)
        self.assertEqual(sum(p.numel() for p in swiglu.parameters()), 24576)
        self.assertEqual(gelu(torch.randn(2, 5, 64)).shape, (2, 5, 64))

    def test_shared_experts_balancing_and_router_gradients(self):
        moe = TopKMoE(8, 12, 4, 2, shared_experts=1).double()
        x = torch.randn(2, 5, 8, dtype=torch.float64)
        output, route = moe(x)
        direct = sum(moe.shared_experts[i](x) for i in range(len(moe.shared_experts)))
        for expert in range(4):
            for slot in range(2):
                selected = route.indices[..., slot] == expert
                direct[selected] += route.weights[..., slot][selected, None] * moe.experts[expert](x[selected])
        torch.testing.assert_close(output, direct)
        loss = output.square().mean() + .01 * route.auxiliary_loss
        loss.backward()
        self.assertGreater(float(moe.router.weight.grad.abs().sum()), 0.)
        old = moe.selection_bias.clone()
        moe.update_selection_bias(torch.tensor([8, 0, 0, 0]), .1)
        torch.testing.assert_close(moe.selection_bias - old, torch.tensor([-.1, .1, .1, .1], dtype=torch.float64))

    def test_selection_bias_changes_selection_but_not_original_gate(self):
        moe = TopKMoE(1, 2, 2, 1).double()
        with torch.no_grad():
            moe.router.weight.copy_(torch.tensor([[.7], [.6]], dtype=torch.float64))
            moe.selection_bias.copy_(torch.tensor([-.2, .2], dtype=torch.float64))
        _, route = moe(torch.ones(1, 1, dtype=torch.float64))
        self.assertEqual(route.indices.item(), 1)
        self.assertEqual(route.weights.item(), 1.)
        torch.testing.assert_close(route.probabilities, torch.tensor([[.52497918747894, .47502081252106]], dtype=torch.float64))

    def test_multi_token_masks_documents_and_trains_all_heads(self):
        heads = MultiTokenHeads(8, 11, horizons=2)
        x = torch.randn(1, 5, 8)
        tokens = torch.tensor([[1, 2, 3, 4, 5]])
        documents = torch.tensor([[0, 0, 1, 1, 1]])
        loss, counts = multitoken_loss(heads(x), tokens, documents, weights=(1., .5))
        self.assertEqual(counts, (3, 1))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in heads.parameters()))
        logits = heads(x)
        expected = F.cross_entropy(torch.stack((logits[0][0, 0], logits[0][0, 2], logits[0][0, 3])), torch.tensor([2, 4, 5]))
        expected += .5 * F.cross_entropy(logits[1][0, 2:3], torch.tensor([5]))
        torch.testing.assert_close(loss, expected)

    def test_online_softmax_is_exact_across_masked_tiles_and_backward(self):
        q, k, v = [torch.randn(2, 3, 7, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        mask = causal_window_mask(torch.arange(7), 7, window=3)
        expected = masked_attention(q, k, v, mask)
        for block in (1, 2, 4, 8):
            actual = online_attention(q, k, v, mask, block)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        expected_grad = torch.autograd.grad(expected.square().sum(), (q, k, v), retain_graph=True)
        actual_grad = torch.autograd.grad(actual.square().sum(), (q, k, v))
        for left, right in zip(expected_grad, actual_grad):
            torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)

    def test_selection_bias_update_invalidates_whole_model_cache(self):
        model = build_model(preset('pico-moe')).eval()
        ids = torch.tensor([[1, 2, 3]])
        _, cache = model.forward_cached(ids[:, :2])
        model.blocks[0].ffn.update_selection_bias(torch.tensor([4, 0, 0, 0]), .1)
        with self.assertRaisesRegex(ValueError, 'changed parameters'):
            model.forward_cached(ids[:, 2:], cache)

    def test_bfloat16_attention_uses_finite_float32_reductions(self):
        q, k, v = [torch.randn(1, 2, 5, 4, dtype=torch.bfloat16) for _ in range(3)]
        mask = causal_window_mask(torch.arange(5), 5)
        expected = masked_attention(q.float(), k.float(), v.float(), mask)
        actual = online_attention(q, k, v, mask, 2)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual.float(), expected, rtol=.01, atol=.01)

    def test_parameter_estimator_matches_every_teaching_preset(self):
        from architecture_lab import PRESETS
        for name, spec in PRESETS.items():
            if spec.scale == 'teaching':
                with self.subTest(preset=name):
                    self.assertEqual(parameter_count(spec), sum(p.numel() for p in build_model(spec).parameters()))
        self.assertEqual(ring_allreduce_bytes(8 * 1024 * 1024, 4), 12 * 1024 * 1024)
        self.assertEqual(ring_allreduce_bytes(123, 1), 0.)
        with self.assertRaises(ValueError):
            ring_allreduce_bytes(123, 0)


if __name__ == '__main__':
    unittest.main()
