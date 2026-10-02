"""Oracle tests: each operator is compared with an independent explicit reference.

The references are written out here (per-head loops, torch SDPA, a transcription of the
YaRN reference code, a naive concatenated MLA). transformers is used only as an extra
oracle and its tests are skipped when it is absent or its API differs.
"""

import gc
import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from architecture_lab import build_model, preset
from architecture_lab.attention import LatentAttention
from architecture_lab.labs import parameter_count
from architecture_lab.model import CausalAttention, apply_rope
from architecture_lab.moe import TopKMoE
from architecture_lab.position import YarnRoPE


def reference_yarn_frequencies(dim, base, factor, original, beta_fast=32.0, beta_slow=1.0):
    """Transcription of find_correction_range, linear_ramp_mask and the blend in YaRN's code."""
    def correction_dim(rotations):
        return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    ramp = ((torch.arange(dim // 2, dtype=torch.float64) - low) / (high - low)).clamp(0, 1)
    extrapolation_factor = 1 - ramp
    return (inv_freq_interpolation * (1 - extrapolation_factor)
            + inv_freq_extrapolation * extrapolation_factor)


def huggingface_yarn(head_width, original, factor, base):
    """(inv_freq, attention_factor) from transformers' YaRN init, or skip when unavailable."""
    try:
        from transformers import LlamaConfig
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        parameters = {"rope_type": "yarn", "factor": factor, "beta_fast": 32.0, "beta_slow": 1.0,
                      "original_max_position_embeddings": original}
        try:
            config = LlamaConfig(hidden_size=4 * head_width, num_attention_heads=4, head_dim=head_width,
                                 max_position_embeddings=int(original * factor),
                                 rope_parameters={**parameters, "rope_theta": base})
        except TypeError:  # transformers 4.x spelling
            config = LlamaConfig(hidden_size=4 * head_width, num_attention_heads=4, head_dim=head_width,
                                 max_position_embeddings=int(original * factor), rope_theta=base,
                                 rope_scaling=parameters)
        return ROPE_INIT_FUNCTIONS["yarn"](config)
    except Exception as error:  # missing package or a different API
        raise unittest.SkipTest(f"transformers YaRN oracle unavailable: {type(error).__name__}")


class AttentionOracleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(1)

    def test_causal_attention_matches_per_head_loop_and_torch_sdpa(self):
        # The loop fixes the grouping: query head h reads KV head h // groups, so neighbouring
        # query heads share one KV head (repeat_interleave). tensor.repeat would give h % kv_heads.
        for name in ("pico-dense", "pico-gqa", "pico-mqa", "llama3-8b-tiny"):
            with self.subTest(name=name):
                spec = preset(name)
                attention = CausalAttention(spec).double()
                x = torch.randn(2, 6, spec.width, dtype=torch.float64)
                positions = torch.arange(6)
                actual, _ = attention(x, positions)

                q = attention._split(attention.q_proj(x), spec.heads, spec.head_width)
                k = attention._split(attention.k_proj(x), spec.kv_heads, spec.head_width)
                v = attention._split(attention.v_proj(x), spec.kv_heads, spec.head_width)
                q, k = apply_rope(q, positions, spec.rope_base), apply_rope(k, positions, spec.rope_base)
                groups = spec.heads // spec.kv_heads
                kv_of_head = [head // groups for head in range(spec.heads)]
                allowed = torch.ones(6, 6, dtype=torch.bool).tril()
                per_head = []
                for head, kv in enumerate(kv_of_head):
                    scores = q[:, head] @ k[:, kv].transpose(-2, -1) / math.sqrt(spec.head_width)
                    per_head.append(scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ v[:, kv])
                expected = attention.o_proj(torch.stack(per_head, 1).transpose(1, 2).reshape(2, 6, spec.width))
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

                # torch's default SDPA scale is 1/sqrt(head_width)
                sdpa = F.scaled_dot_product_attention(q, k[:, kv_of_head], v[:, kv_of_head], is_causal=True)
                sdpa = attention.o_proj(sdpa.transpose(1, 2).reshape(2, 6, spec.width))
                torch.testing.assert_close(actual, sdpa, rtol=1e-12, atol=1e-12)


class YarnOracleTests(unittest.TestCase):
    CASES = ((16, 4096, 4.0, 10_000.0), (16, 128, 4.0, 10_000.0), (64, 2048, 2.5, 10_000.0),
             (128, 4096, 4.0, 10_000.0), (128, 8192, 8.0, 500_000.0))

    def test_default_frequencies_match_the_reference_dimension_ramp(self):
        for width, original, scale, base in self.CASES:
            with self.subTest(width=width, original=original, scale=scale):
                actual = YarnRoPE(width, original, scale, base=base).frequencies()
                torch.testing.assert_close(
                    actual, reference_yarn_frequencies(width, base, scale, original), rtol=1e-12, atol=0)

    def test_default_frequencies_match_transformers(self):
        for width, original, scale, base in self.CASES:
            with self.subTest(width=width, original=original, scale=scale):
                inv_freq, attention_factor = huggingface_yarn(width, original, scale, base)
                actual = YarnRoPE(width, original, scale, base=base)
                torch.testing.assert_close(actual.frequencies().float(), inv_freq.float(), rtol=1e-6, atol=0)
                self.assertAlmostEqual(actual.attention_multiplier, attention_factor, places=12)

    def test_rotation_ramp_is_the_papers_equation_and_differs_between_the_bounds(self):
        width, original, scale, base = 16, 4096, 4.0, 10_000.0
        rope = YarnRoPE(width, original, scale, base=base, ramp="rotations")
        theta = base ** (-2 * torch.arange(width // 2, dtype=torch.float64) / width)
        gamma = ((original * theta / (2 * math.pi) - 1.0) / (32.0 - 1.0)).clamp(0, 1)
        torch.testing.assert_close(rope.frequencies(), theta * ((1 - gamma) / scale + gamma))
        dimension = YarnRoPE(width, original, scale, base=base).frequencies()
        self.assertEqual(float(dimension[0]), float(rope.frequencies()[0]))
        self.assertEqual(float(dimension[-1]), float(rope.frequencies()[-1]))
        self.assertGreater(float((dimension / rope.frequencies() - 1).abs().max()), 0.1)
        with self.assertRaises(ValueError):
            YarnRoPE(width, original, scale, ramp="linear")

    def test_attention_multiplier_is_one_plus_point_one_log_scale(self):
        self.assertAlmostEqual(YarnRoPE(16, 4096, 4.0).attention_multiplier, 1.13862944, places=8)
        self.assertAlmostEqual(YarnRoPE(16, 4096, 8.0).attention_multiplier, 1.20794415, places=8)
        self.assertEqual(YarnRoPE(16, 4096, 1.0).attention_multiplier, 1.0)

    def test_rotation_matches_hand_coded_pairs_and_multiplier(self):
        width, original, scale = 16, 128, 4.0
        rope = YarnRoPE(width, original, scale)
        x = torch.randn(1, 2, 5, width, dtype=torch.float64)
        positions = torch.arange(5)
        theta = reference_yarn_frequencies(width, 10_000.0, scale, original)
        expected = torch.empty_like(x)
        for pair in range(width // 2):
            angle = positions.double() * theta[pair]
            even, odd = x[..., 2 * pair], x[..., 2 * pair + 1]
            expected[..., 2 * pair] = even * angle.cos() - odd * angle.sin()
            expected[..., 2 * pair + 1] = even * angle.sin() + odd * angle.cos()
        torch.testing.assert_close(rope(x, positions), expected * (1 + 0.1 * math.log(scale)))
        bare = YarnRoPE(width, original, scale, apply_multiplier=False)
        torch.testing.assert_close(bare(x, positions), expected)


class LatentAttentionOracleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        torch.set_num_threads(1)

    @staticmethod
    def naive_concat(model, x):
        """Concatenate content and positional parts into ordinary Q/K and call torch SDPA."""
        batch, length, _ = x.shape
        positions = torch.arange(length)
        heads, c, r = model.heads, model.content_width, model.positional_width
        query_latent = model.q_down(x)
        latent = model.kv_down(x)
        if model.q_norm is not None:
            query_latent, latent = model.q_norm(query_latent), model.kv_norm(latent)
        q_content = model.q_content(query_latent).view(batch, length, heads, c).transpose(1, 2)
        q_position = model.q_position(query_latent).view(batch, length, heads, r).transpose(1, 2)
        k_content = model.k_content(latent).view(batch, length, heads, c).transpose(1, 2)
        value = model.v_content(latent).view(batch, length, heads, c).transpose(1, 2)
        k_position = model.k_position(x).unsqueeze(1)
        logit_scale = 1.0
        if model.yarn is None:
            q_position = apply_rope(q_position, positions, model.rope_base)
            k_position = apply_rope(k_position, positions, model.rope_base)
        else:
            rope = YarnRoPE(r, model.yarn.original_context, model.yarn.scale, base=model.rope_base,
                            apply_multiplier=model.mscale_scope == "positional")
            q_position, k_position = rope(q_position, positions), rope(k_position, positions)
            if model.mscale_scope == "logit":
                logit_scale = (1 + 0.1 * math.log(model.yarn.scale)) ** 2
        query = torch.cat((q_content, q_position), dim=-1)
        key = torch.cat((k_content, k_position.expand(-1, heads, -1, -1)), dim=-1)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True,
                                                  scale=logit_scale / math.sqrt(c + r))
        return model.output(attended.transpose(1, 2).reshape(batch, length, heads * c))

    def configurations(self):
        base = {"width": 16, "heads": 4, "content_width": 4, "positional_width": 4, "kv_rank": 6, "query_rank": 8}
        yarn = {"yarn_original_context": 4, "yarn_scale": 4}
        yield "plain", base
        yield "yarn positional scale", {**base, **yarn}
        yield "yarn whole-logit mscale^2", {**base, **yarn, "mscale_scope": "logit"}
        yield "latent norm", {**base, "latent_norm": True}
        yield "latent norm, yarn, whole-logit", {**base, **yarn, "latent_norm": True, "mscale_scope": "logit"}

    def test_matches_naive_concatenation_and_cache_equals_full_forward(self):
        x = torch.randn(2, 9, 16, dtype=torch.float64)
        for label, options in self.configurations():
            for mode in ("reconstruct", "absorbed"):
                with self.subTest(configuration=label, mode=mode):
                    model = LatentAttention(**options, inference_mode=mode).double().eval()
                    if model.q_norm is not None:  # non-trivial gains, so the norm cannot hide as identity
                        with torch.no_grad():
                            model.q_norm.weight.uniform_(0.5, 1.5)
                            model.kv_norm.weight.uniform_(0.5, 1.5)
                    with torch.no_grad():
                        full, _ = model(x)
                        torch.testing.assert_close(full, self.naive_concat(model, x), rtol=1e-12, atol=1e-12)
                        pieces, cache = [], None
                        for start, stop in ((0, 3), (3, 6), (6, 9)):
                            piece, cache = model(x[:, start:stop], cache, use_cache=True)
                            pieces.append(piece)
                        torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=1e-12, atol=1e-13)
                        self.assertLess(float((torch.cat(pieces, dim=1) - full).abs().max()), 1e-14)

    def test_mscale_scope_changes_only_what_the_reference_scales(self):
        options = {"width": 16, "heads": 4, "content_width": 4, "positional_width": 4, "kv_rank": 6,
                   "query_rank": 8, "yarn_original_context": 4, "yarn_scale": 4}
        positional = LatentAttention(**options).double().eval()
        logit = LatentAttention(**options, mscale_scope="logit").double().eval()
        logit.load_state_dict(positional.state_dict())
        x = torch.randn(1, 7, 16, dtype=torch.double)
        with torch.no_grad():
            self.assertGreater(float((positional(x)[0] - logit(x)[0]).abs().max()), 1e-4)
        with self.assertRaises(ValueError):
            LatentAttention(16, 4, 4, 4, 6, 8, mscale_scope="logit")  # needs YaRN
        with self.assertRaises(ValueError):
            LatentAttention(16, 4, 4, 4, 6, 8, mscale_scope="all")

    def test_latent_norm_and_whole_logit_mscale_match_transformers_deepseek_v3_attention(self):
        heads, c, r, kv_rank, query_rank, width, original, factor, length = 4, 8, 4, 6, 10, 32, 16, 4.0, 12
        model = LatentAttention(width, heads, c, r, kv_rank, query_rank, yarn_original_context=original,
                                yarn_scale=factor, latent_norm=True, norm_eps=1e-6,
                                mscale_scope="logit").double().eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(0, 0.5)
            x = torch.randn(2, length, width, dtype=torch.float64)
            try:  # the module API moves between transformers releases; skip rather than fail on drift
                from transformers import DeepseekV3Config
                from transformers.models.deepseek_v3 import modeling_deepseek_v3 as hf_module
                config = DeepseekV3Config(
                    hidden_size=width, num_attention_heads=heads, num_key_value_heads=heads,
                    q_lora_rank=query_rank, kv_lora_rank=kv_rank, qk_rope_head_dim=r, qk_nope_head_dim=c,
                    v_head_dim=c, max_position_embeddings=int(original * factor), rms_norm_eps=1e-6,
                    rope_interleave=True, num_hidden_layers=1, attn_implementation="eager",
                    rope_parameters={"rope_type": "yarn", "factor": factor, "rope_theta": 10_000.0,
                                     "original_max_position_embeddings": original, "beta_fast": 32.0,
                                     "beta_slow": 1.0, "mscale": 1.0, "mscale_all_dim": 1.0})
                reference = hf_module.DeepseekV3Attention(config, 0).double().eval()
                rotary = hf_module.DeepseekV3RotaryEmbedding(config).double()
                per_head = lambda a, b, wb: torch.cat(  # Hugging Face fuses [content; second] rows head by head
                    [torch.cat((a[h * c:(h + 1) * c], b[h * wb:(h + 1) * wb]), 0) for h in range(heads)], 0)
                reference.q_a_proj.weight.copy_(model.q_down.weight)
                reference.q_a_layernorm.weight.copy_(model.q_norm.weight)
                reference.q_b_proj.weight.copy_(per_head(model.q_content.weight, model.q_position.weight, r))
                reference.kv_a_proj_with_mqa.weight.copy_(torch.cat((model.kv_down.weight, model.k_position.weight), 0))
                reference.kv_a_layernorm.weight.copy_(model.kv_norm.weight)
                reference.kv_b_proj.weight.copy_(per_head(model.k_content.weight, model.v_content.weight, c))
                reference.o_proj.weight.copy_(model.output.weight)
                cos, sin = rotary(x, torch.arange(length)[None].expand(2, -1))
                mask = torch.full((length, length), float("-inf"), dtype=torch.float64).triu(1)[None, None]
                expected, _ = reference(x, (cos, sin), mask)
            except Exception as error:
                raise unittest.SkipTest(f"transformers DeepSeek-V3 oracle unavailable: {type(error).__name__}")
            actual, _ = model(x)
            # transformers builds its cos/sin tables in float32, hence the loose tolerance
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5 * float(expected.abs().max()))
            positional = LatentAttention(width, heads, c, r, kv_rank, query_rank, yarn_original_context=original,
                                         yarn_scale=factor, latent_norm=True, norm_eps=1e-6).double().eval()
            positional.load_state_dict(model.state_dict())
            self.assertGreater(float((positional(x)[0] - expected).abs().max()), 0.1)  # the old scope differs

    def test_default_has_no_latent_norm_parameters_and_spec_options_count_them(self):
        plain = LatentAttention(16, 4, 4, 4, kv_rank=6, query_rank=8)
        self.assertIsNone(plain.q_norm)
        self.assertEqual(len(list(plain.parameters())), 8)
        spec = replace(preset("pico-mla"), mla_latent_norm=True)
        model = build_model(spec)
        self.assertEqual(parameter_count(spec), sum(p.numel() for p in model.parameters()))
        self.assertEqual(parameter_count(spec) - parameter_count(preset("pico-mla")),
                         spec.layers * (spec.mla_query_rank + spec.mla_kv_rank))
        logit = replace(preset("pico-mla-yarn-moe"), mla_mscale_scope="logit", mla_latent_norm=True)
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        decoder = build_model(logit).eval()
        full = decoder(ids)
        first, cache = decoder.forward_cached(ids[:, :4])
        last, _ = decoder.forward_cached(ids[:, 4:], cache)
        torch.testing.assert_close(full, torch.cat((first, last), dim=1), rtol=1e-5, atol=1e-5)


class MoeOracleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(44)
        torch.set_num_threads(1)

    def test_auxiliary_loss_is_experts_times_assignment_fraction_dot_mean_probability(self):
        experts, top_k, tokens = 4, 2, 6
        moe = TopKMoE(8, 12, experts, top_k).double()
        x = torch.randn(1, tokens, 8, dtype=torch.float64)
        _, routing = moe(x)
        logits = moe.router(x.reshape(tokens, 8)).detach().tolist()
        counts, mean_probability = [0] * experts, [0.0] * experts
        for row in logits:
            normaliser = sum(math.exp(value) for value in row)
            for expert in range(experts):
                mean_probability[expert] += math.exp(row[expert]) / normaliser / tokens
            for expert in sorted(range(experts), key=lambda e: -row[e])[:top_k]:
                counts[expert] += 1
        expected = experts * sum(count / (tokens * top_k) * p for count, p in zip(counts, mean_probability))
        self.assertEqual(routing.counts.tolist(), counts)
        self.assertAlmostEqual(float(routing.auxiliary_loss.detach()), expected, places=12)

    def test_balanced_routing_gives_exactly_one(self):
        moe = TopKMoE(4, 6, 4, 1).double()
        with torch.no_grad():
            moe.router.weight.copy_(5 * torch.eye(4, dtype=torch.float64))
        _, routing = moe(torch.eye(4, dtype=torch.float64))
        self.assertEqual(routing.counts.tolist(), [1, 1, 1, 1])
        self.assertAlmostEqual(float(routing.auxiliary_loss), 1.0, places=12)
        collapsed = moe(torch.eye(4, dtype=torch.float64)[:1].repeat(4, 1))[1]
        self.assertGreater(float(collapsed.auxiliary_loss), 1.5)

    def test_auxiliary_loss_relates_to_the_mixtral_formulation_by_top_k(self):
        # Mixtral's load_balancing_loss_func divides counts by tokens, not assignments, so it is
        # top_k times the value here (they coincide for top_k = 1).
        try:
            from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func
        except Exception as error:
            raise unittest.SkipTest(f"transformers oracle unavailable: {type(error).__name__}")
        for top_k in (1, 2):
            with self.subTest(top_k=top_k):
                moe = TopKMoE(8, 12, 4, top_k).double()
                x = torch.randn(1, 10, 8, dtype=torch.float64)
                _, routing = moe(x)
                try:
                    reference = load_balancing_loss_func((moe.router(x.reshape(10, 8)),), 4, top_k)
                except Exception as error:
                    raise unittest.SkipTest(f"transformers API differs: {type(error).__name__}")
                self.assertAlmostEqual(float(reference.detach()), top_k * float(routing.auxiliary_loss.detach()), places=5)

    def test_top_one_gives_the_router_no_task_gradient_unless_gate_uses_router_probability(self):
        x = torch.randn(2, 5, 8, dtype=torch.float64)
        for top_k in (1, 2):
            moe = TopKMoE(8, 12, 4, top_k).double()
            output, routing = moe(x)
            output.square().mean().backward()
            norm = float(moe.router.weight.grad.abs().sum())
            if top_k == 1:
                self.assertTrue(torch.equal(routing.weights, torch.ones_like(routing.weights)))
                self.assertEqual(norm, 0.0)
                moe.zero_grad()
                _, routing = moe(x)
                routing.auxiliary_loss.backward()  # only the balance loss trains a top-1 router here
                self.assertGreater(float(moe.router.weight.grad.abs().sum()), 0.0)
            else:
                self.assertGreater(norm, 0.0)
        switch = TopKMoE(8, 12, 4, 1, gate="router_probability").double()
        output, routing = switch(x)
        torch.testing.assert_close(routing.weights.squeeze(-1), routing.probabilities.max(-1).values)
        output.square().mean().backward()
        self.assertGreater(float(switch.router.weight.grad.abs().sum()), 0.0)
        with self.assertRaises(ValueError):
            TopKMoE(8, 12, 4, 1, gate="hard")


class DecoderRoutingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(45)
        torch.set_num_threads(1)

    def test_routing_is_available_without_changing_the_default_return(self):
        spec = preset("pico-moe")
        model = build_model(spec)
        ids = torch.randint(0, spec.vocab_size, (2, 7))
        logits = model(ids)
        self.assertIsInstance(logits, torch.Tensor)
        with_routing, routing = model(ids, return_routing=True)
        torch.testing.assert_close(logits, with_routing)
        self.assertEqual(len(routing.layers), spec.layers)
        self.assertEqual([int(c.sum()) for c in routing.counts], [2 * 7 * spec.moe_top_k] * spec.layers)
        torch.testing.assert_close(routing.auxiliary_loss, sum(layer.auxiliary_loss for layer in routing.layers))
        (logits.sum() * 0 + routing.auxiliary_loss).backward()  # the balance loss reaches every router
        for block in model.blocks:
            self.assertGreater(float(block.ffn.router.weight.grad.abs().sum()), 0.0)

    def test_dense_decoder_reports_no_routing(self):
        model = build_model(preset("pico-dense"))
        _, routing = model(torch.tensor([[1, 2, 3]]), return_routing=True)
        self.assertEqual(routing.layers, (None, None))
        self.assertEqual(float(routing.auxiliary_loss), 0.0)

    def test_selection_bias_hook_moves_each_block_from_its_own_counts(self):
        model = build_model(preset("pico-moe")).eval()
        _, routing = model(torch.tensor([[1, 2, 3, 4, 5]]), return_routing=True)
        model.update_selection_biases(routing, 0.25)
        for block, record in zip(model.blocks, routing.layers):
            counts = record.counts.float()
            torch.testing.assert_close(block.ffn.selection_bias, 0.25 * torch.sign(counts.mean() - counts))
        self.assertGreater(float(sum(b.ffn.selection_bias.abs().sum() for b in model.blocks)), 0.0)

    def test_top_one_decoder_can_use_the_router_probability_gate(self):
        base = replace(preset("pico-moe"), name="pico-moe-top1", moe_top_k=1)
        ids = torch.tensor([[1, 2, 3, 4]])
        plain = build_model(base)
        plain(ids).square().mean().backward()
        self.assertEqual(float(plain.blocks[0].ffn.router.weight.grad.abs().sum()), 0.0)
        switch = build_model(replace(base, moe_gate="router_probability"))
        switch(ids).square().mean().backward()
        self.assertGreater(float(switch.blocks[0].ffn.router.weight.grad.abs().sum()), 0.0)


class CacheOwnerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(47)

    def test_cache_from_a_freed_model_is_rejected_by_a_later_model(self):
        spec = preset("pico-dense")
        ids = torch.tensor([[1, 2, 3, 4]])
        for _ in range(20):
            first = build_model(spec).eval()
            _, cache = first.forward_cached(ids[:, :2])
            del first
            gc.collect()
            second = build_model(spec).eval()  # CPython often reuses the freed model's address
            with self.assertRaisesRegex(ValueError, "another model"):
                second.forward_cached(ids[:, 2:], cache)

    def test_cache_owner_does_not_depend_on_object_identity_numbers(self):
        spec = preset("pico-dense")
        ids = torch.tensor([[1, 2, 3, 4]])
        first, second = build_model(spec).eval(), build_model(spec).eval()
        with patch("architecture_lab.model.id", lambda obj: 7, create=True):  # force every id to collide
            _, cache = first.forward_cached(ids[:, :2])
            with self.assertRaisesRegex(ValueError, "another model"):
                second.forward_cached(ids[:, 2:], cache)
            _, own = first.forward_cached(ids[:, 2:], cache)  # the owner still accepts its own cache
        self.assertEqual(own.tokens.shape[1], 4)

    def test_a_deep_copy_does_not_inherit_the_cache(self):
        import copy
        model = build_model(preset("pico-dense")).eval()
        ids = torch.tensor([[1, 2, 3]])
        _, cache = model.forward_cached(ids[:, :2])
        with self.assertRaisesRegex(ValueError, "another model"):
            copy.deepcopy(model).forward_cached(ids[:, 2:], cache)


if __name__ == "__main__":
    unittest.main()
