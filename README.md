# LLM architecture lab

This is the student concept project for course lessons **31–50**. It starts
from a tiny decoder resembling the course's PicoLLM, then changes the number
of stored key/value heads through a versioned `ModelSpec`. The current
implementation includes adjacent-pair RoPE, pre-RMSNorm, SwiGLU, explicit
causal MHA/MQA/GQA, and a grouped KV cache. It is a reference implementation
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
```

If PyTorch is unavailable, the dependency-free specification tests still run
with `PYTHONPATH=src python3 -m unittest discover -s tests -v`; model tests
report `skipped` until PyTorch is installed. `python3 -m compileall -q src tests`
performs syntax checking without PyTorch.

## Change the model, keep the calling code

```python
from architecture_lab import build_model, preset

spec = preset("pico-gqa")  # or pico-dense, pico-mqa, llama3-8b-tiny
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
parameters. A cache belongs to one unchanged model and token prefix. These
are numerical and structural tests, not performance measurements.

## Independent mechanism references

These modules are **not yet wired into `ModelSpec` or `DecoderLM` presets**.
They are small, importable experiments for lessons 34, 36, and 44. Run
`PYTHONPATH=src python -m unittest discover -s tests -v` after installing
PyTorch. The tests include forward/backward, causal, cache, frequency, routing
and weight-conservation checks on CPU.

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

`YarnRoPE` implements a **fixed** scale: the paper's NTK-by-parts frequency
ramp using rotations in the original context, plus its attention multiplier
on both Q and K. It does not implement Dynamic-YaRN, fine-tuning, or a
length-extrapolation evaluation. Fix the scale for a cached decode; changing
it mid-cache would require re-rotating earlier keys. The paper's suggested
`alpha=1`, `beta=32`, and attention formula were fitted for LLaMA-family
experiments, not validated here as universal settings.

`LatentAttention` implements low-rank Q and joint KV projections, a shared
decoupled RoPE key, causal masking, and a cache of KV latents plus positional
keys. For clarity it **reconstructs** content K and V from the whole cached
latent on each call. It does not implement DeepSeek-V2's absorbed inference
weights, fused kernels, full architecture, checkpoint compatibility, or
published KV reduction measurements. Its standalone `LatentCache` has no
model-owner/parameter-version guard; use it only with the same unchanged
module and exact token prefix.

`TopKMoE` computes selected-expert softmax weights, normalizes them to one
per token, and dispatches each token to `top_k` SwiGLU experts. No token is
dropped. It has no shared expert, capacity limit, distributed all-to-all,
load-balancing loss, or router stability training recipe. With `top_k=1`,
normalizing only the selected logit makes the gate weight exactly one and
gives the router no task-loss gradient; use `top_k=2` for this gradient demo.
Top-k selection itself is discrete, so gradients flow through selected
logits, not through the choice of expert index.

## Scope and next lessons

Fixed-scale YaRN, a readable MLA core, and sparse top-k dispatch now have
standalone CPU references. Sliding/sparse/linear attention, shared-expert
balancing, FlashAttention, FP8 and distributed execution remain future work.
See [lesson map](LESSON_MAP.md). No benchmark or published-model parity is
claimed here.

## Primary sources

- Su et al., [RoFormer / RoPE](https://arxiv.org/abs/2104.09864): rotary
  position treatment. This lab chooses adjacent coordinate pairs explicitly.
- Ainslie et al., [GQA](https://arxiv.org/abs/2305.13245): grouped query
  attention and the MHA↔MQA spectrum.
- Shazeer, [One Write-Head Is All You Need](https://arxiv.org/abs/1911.02150):
  multi-query decoding and cache motivation.
- Meta, [Llama 3 model SKU definitions](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py)
  and [model code](https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py):
  published 8B dimensions and SwiGLU width derivation. The 8K context is
  stated in Meta's [model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3/MODEL_CARD.md).
- Peng et al., [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/pdf/2309.00071), sections 3.2–3.4:
  NTK-by-parts frequency ramp, attention scaling, and the dynamic-cache caveat.
- DeepSeek-AI, [DeepSeek-V2](https://arxiv.org/pdf/2405.04434), section 2.1:
  low-rank KV compression and decoupled positional key. This reference omits
  its inference weight absorption.
- Lepikhin et al., [GShard](https://arxiv.org/abs/2006.16668): sparse expert
  selection and conditional computation motivation. This router is a small
  teaching variant, not a reproduction of its sharding or capacity policy.

No external model weights or datasets are bundled.
