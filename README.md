# LLM architecture lab

This is the student concept project for course lessons **31–50**. It starts
from a tiny decoder resembling the course's PicoLLM, then selects attention,
position, and FFN mechanisms through a versioned `ModelSpec`. The current
implementation includes adjacent-pair RoPE, fixed-scale YaRN, pre-RMSNorm,
SwiGLU or top-k MoE, causal MHA/MQA/GQA or MLA, and their caches.
It is a reference implementation
for understanding shapes and correctness; it is not a fast training kernel.

## Start

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python -m architecture_lab.checks --preset pico-dense
python -m architecture_lab.checks --preset pico-gqa
python -m architecture_lab.checks --preset pico-mqa
python -m architecture_lab.checks --preset pico-yarn
python -m architecture_lab.checks --preset pico-mla
python -m architecture_lab.checks --preset pico-mla-absorbed
python -m architecture_lab.checks --preset pico-mla-yarn-moe
python -m architecture_lab.checks --preset pico-moe
```

If PyTorch is unavailable, the dependency-free specification tests still run
with `PYTHONPATH=src python3 -m unittest discover -s tests -p test_spec.py -v`.
`python3 -m compileall -q src tests` performs syntax checking without PyTorch.

## Change the model, keep the calling code

```python
from architecture_lab import build_model, preset

spec = preset("pico-gqa")  # change to any teaching preset listed by checks --help
model = build_model(spec)
print(spec.to_dict(), spec.attention_kind)
```

`ModelSpec` is frozen and carries `schema_version=1`; `to_dict`/`from_dict`
provide a simple format for experiment manifests. Invalid widths, group
ratios, RoPE dimensions, and version numbers are rejected. Changing a preset
creates a **new randomly initialized model**. Checkpoints and optimizer state
do not automatically transfer between variants. The lesson should compare
them under a recorded seed, dataset, token budget, parameter count, and
evaluation protocol. The cache stores `kv_heads` rather than repeating to
`heads`; `kv_cache_bytes` gives only the theoretical K/V tensor payload.

`llama3-8b-reference` records selected published-size dimensions but
`build_model` explicitly refuses to allocate it. `llama3-8b-tiny` preserves
the 4-to-1 query/KV grouping and the broad decoder pattern at teaching scale;
it has a different vocabulary, width, layer count, context and FFN width. It
does **not** reproduce Llama 3's tokenizer, training recipe, weights, quality,
throughput, or full implementation details. See [lesson map](LESSON_MAP.md).

## Correctness contract

The executable checks compare full-sequence logits to three cached chunks,
verify cache shapes, check RoPE norm and relative-position identities, ensure
future tokens cannot change earlier outputs, and backpropagate through all
parameters. `tests/test_oracles.py` adds independent references for the attention
scale and GQA head grouping (a per-head loop and torch SDPA), the YaRN frequencies and
multiplier, MLA against a naive concatenated Q/K, and the MoE balance-loss factor.
A cache belongs to one unchanged model instance and token prefix: the owner is a
per-instance sentinel compared by identity (a freed model's `id` can be reused by a
later one, a deep copy gets its own sentinel), plus a parameter-version check. These
are numerical and structural tests, not performance measurements.

## Selectable mechanism references

`pico-yarn`, `pico-mla`, and `pico-moe` select these same operators through
`ModelSpec` and `DecoderLM`. They are small, importable experiments for lessons
34, 36, and 44. Run
`PYTHONPATH=src python -m unittest discover -s tests -v` after installing
PyTorch. The tests include whole-model cached/full parity, forward/backward,
causal, cache, frequency, routing and weight-conservation checks on CPU.
`ModelSpec` rejects incomplete mechanism settings; its original fields remain
the default, so existing version-1 spec dictionaries still load. An MLA spec
uses `kv_heads=heads` as a reserved legacy field, while `kv_cache_bytes` counts
its shared latent and positional key. A custom spec can combine MLA and MoE
without changing the decoder's logits API. `pico-mla-absorbed` swaps only the
cached inference algebra. `pico-mla-yarn-moe` combines fixed-scale YaRN on
MLA's decoupled positional path with sparse routed experts. Both remain small
teaching models rather than DeepSeek-V2 reproductions.

```python
import torch
from architecture_lab.position import YarnRoPE
from architecture_lab.attention import LatentAttention
from architecture_lab.moe import TopKMoE

rope = YarnRoPE(head_width=16, original_context=4096, scale=4)
q = torch.randn(1, 4, 8, 16)
rotated_q = rope(q, torch.arange(8))

mla = LatentAttention(width=16, heads=4, content_width=4,
                      positional_width=4, kv_rank=6, query_rank=8).eval()
x = torch.randn(1, 8, 16)
full, _ = mla(x)
first, cache = mla(x[:, :4], use_cache=True)
last, cache = mla(x[:, 4:], cache, use_cache=True)
torch.testing.assert_close(full, torch.cat((first, last), dim=1))

moe = TopKMoE(width=16, ff_width=32, experts=4, top_k=2)
output, routing = moe(x)
assert routing.counts.sum().item() == 2 * x.shape[0] * x.shape[1]
```

`YarnRoPE` implements a **fixed** scale: NTK-by-parts frequencies plus the paper's
attention multiplier `0.1 ln(s) + 1` on both Q and K. It does not implement
Dynamic-YaRN, fine-tuning, or a length-extrapolation evaluation. Fix the scale for a
cached decode; changing it mid-cache would require re-rotating earlier keys. The
paper's suggested `alpha=1`, `beta=32`, and attention formula were fitted for
LLaMA-family experiments, not validated here as universal settings.

The frequency ramp has two forms. The default, `ramp="dimension"`, is the one in the
authors' released code and in Hugging Face `transformers`
(`_compute_yarn_parameters`): the extrapolation weight falls linearly in the pair index
between `low = floor(d(beta))` and `high = ceil(d(alpha))`, where
`d(r) = D ln(L / (2 pi r)) / (2 ln b)` is the pair that completes `r` turns over the
original context `L`. `ramp="rotations"` is the paper's printed equation, linear in the
rotation count `L theta / (2 pi)`. When the ramp saturates at both ends they agree on the
fastest pairs (unchanged) and on the slowest pairs (interpolated by `1/s`) and differ only
between the bounds, which matters when a checkpoint fine-tuned with the reference code is ported. The default
changed from `rotations` to `dimension`; the endpoint frequencies printed by lesson 34
(`first/last frequencies 1.0 7.906e-05`) and the multiplier `1.13862944` are the same
under both. `tests/test_oracles.py` compares the default with a transcription of the
reference code and, when `transformers` is installed, with its YaRN init function.
`apply_multiplier=False` leaves the multiplier to the caller (see MLA below).

`LatentAttention` implements low-rank Q and joint KV projections, a shared
decoupled RoPE key, causal masking, and a cache of KV latents plus positional
keys. The default path reconstructs content K and V from cached latents.
With `mla_inference_mode="absorbed"`, eval-mode cached inference instead
computes `qᵀW_UK c` and `W_O W_UV Σp c`, so neither reconstructed content K
nor V is formed. Double-precision tests compare both paths, including chunked
caches. MLA can apply fixed-scale YaRN to the decoupled positional Q/K only;
the content path remains unrotated. This is an algebraic CPU reference, not a
latency benchmark or optimized kernel. It omits the full DeepSeek-V2
architecture, checkpoint compatibility, and published KV reduction
measurements. Its standalone `LatentCache` has no
model-owner/parameter-version guard; use it only with the same unchanged
module and exact token prefix. The whole `DecoderLM` applies those cache guards.

Two switches align the layer with DeepSeek-V3 as implemented in Hugging Face
`DeepseekV3Attention` and are **off by default**, so parameter counts and the printed
lesson numbers do not change. `latent_norm=True` (`mla_latent_norm` in `ModelSpec`)
adds the `q_a` and `kv_a` RMSNorm on the compressed latents; the cache then stores
the normalised KV latent. `mscale_scope="logit"` (`mla_mscale_scope`, needs YaRN)
rotates Q/K without the YaRN multiplier and multiplies the whole logit by
`(0.1 ln s + 1)^2`, i.e. softmax scale `(c + r)^(-1/2) * mscale^2` over content plus
positional parts, as DeepSeek-V3 does when `rope_scaling` carries an `mscale`. The
default `mscale_scope="positional"` scales only the positional score by the squared
multiplier. When `transformers` is installed, a test compares the layer (latent norm,
YaRN, whole logit) with Hugging Face `DeepseekV3Attention`; they agree to about 1e-7
relative, the float32 precision of Hugging Face's cos/sin tables, and the
positional-only scope does not match it. The test is skipped if the package or its
module API is unavailable.

`TopKMoE` computes selected-expert softmax weights, normalizes them to one
per token, and dispatches each token to `top_k` SwiGLU experts. No token is
dropped. The standalone constructor can add shared experts. Routing returns
an optional balance-loss term and exposes a selection-only bias update; the
caller chooses either training treatment. The decoder defaults keep both
mechanisms inactive. Capacity limits and distributed all-to-all are not implemented.
The selection bias is added to the router **logits** before top-k, not to
post-activation affinities as in DeepSeek-V3's sigmoid routing; gate weights always use
the unbiased scores. The balance loss is `E * sum_e(f_e * P_e)` with `f_e` the share of
assignments (`counts / (tokens * top_k)`), so perfectly balanced routing gives exactly
1.0; Hugging Face's Mixtral `load_balancing_loss_func` divides by tokens instead and
returns `top_k` times this value (equal for `top_k=1`).

With `top_k=1` and the default `gate="selected_softmax"`, normalizing only the selected
logit makes the gate weight exactly one and its derivative exactly zero, so the router
receives **no task-loss gradient** (only the balance loss reaches it); use `top_k=2` for
the gradient demo. `gate="router_probability"` (`moe_gate` in `ModelSpec`) instead
multiplies the selected expert by its full-router probability, Switch Transformer style,
which trains a top-1 router; its weights sum to less than one.
Top-k selection itself is discrete, so gradients flow through selected
logits, not through the choice of expert index.

`DecoderLM(ids, return_routing=True)` returns `(logits, DecoderRouting)`; the default call
still returns the logits tensor. `DecoderRouting.layers` holds one `Routing` (indices,
weights, counts, probabilities, auxiliary loss) per block, `None` for dense blocks;
`.auxiliary_loss` sums the MoE blocks, and `model.update_selection_biases(routing, rate)`
applies the loss-free bias update from each block's own counts. A trainer would add
`alpha * routing.auxiliary_loss` to the task loss. Cached decoding (`forward_cached`)
does not return routing.

## Scope and next lessons

Fixed-scale YaRN, a readable MLA core, and sparse top-k dispatch have
selectable CPU teaching presets. `labs.py` supplies separate differentiable
references for the remaining mechanism lessons: local and selected-block
masks, normalized recurrent kernel attention, QK normalization, explicit
residual placements/gates, matched GELU/SwiGLU budgets, boundary-safe
multi-token heads/loss, online softmax, and exact parameter/cost ledgers.
Run `python -m architecture_lab.checks --labs` for seeded output checkpoints.
`tests/test_labs.py` checks outputs, gradients, state, and causal invariants.

These lab operators are not all exposed as `ModelSpec` decoder presets.
The recurrent rule is ELU+1 normalized kernel attention, not Kimi KDA.
Multi-token heads are independent horizon projections, not DeepSeek's
sequential MTP modules. The online-softmax reference accepts a full mask and
uses Python/autograd operations; it is not a fused FlashAttention kernel.
Sparse masks are checked with a dense oracle, not a sparse GPU kernel.
BF16 checks report numerical error, not a throughput/quality comparison.
FP8, distributed dispatch, published-model checkpoints, long-context
quality evaluations, and GPU performance experiments remain outside these
CPU correctness references.
See [lesson map](LESSON_MAP.md). No benchmark or published-model parity is
claimed here.

## Primary sources

- Su et al., [RoFormer / RoPE](https://arxiv.org/abs/2104.09864): rotary
  position treatment. This lab chooses adjacent coordinate pairs explicitly.
- Ainslie et al., [GQA](https://arxiv.org/abs/2305.13245): grouped query
  attention and the MHA↔MQA spectrum.
- Shazeer, [One Write-Head Is All You Need](https://arxiv.org/abs/1911.02150):
  multi-query decoding and cache motivation.
- Meta, [Llama 3 model SKU definitions](https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/sku_list.py)
  and [model code](https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3/model.py):
  published 8B dimensions and SwiGLU width derivation. The 8K context is
  stated in Meta's [model card](https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3/MODEL_CARD.md).
- Peng et al., [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/pdf/2309.00071), sections 3.2–3.4:
  NTK-by-parts frequency ramp, attention scaling, and the dynamic-cache caveat.
- DeepSeek-AI, [DeepSeek-V2](https://arxiv.org/pdf/2405.04434), section 2.1,
  equations 9–19: low-rank KV compression, decoupled positional key, and
  algebraic inference weight absorption.
- Lepikhin et al., [GShard](https://arxiv.org/abs/2006.16668): sparse expert
  selection and conditional computation motivation. This router is a small
  teaching variant, not a reproduction of its sharding or capacity policy.

No external model weights or datasets are bundled.
