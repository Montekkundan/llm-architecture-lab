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

## Scope and next lessons

YaRN, MLA, sliding/sparse/linear attention, MoE, FlashAttention, FP8 and
distributed execution are **not implemented** in this starter. Their
lessons have file and test targets in `LESSON_MAP.md`; those targets become
complete only after their derivations, operators, backward passes, cache
contracts, source citations and tests are added. No benchmark or published
model parity is claimed here.

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

No external model weights or datasets are bundled. This directory has no
GitHub remote and has not been published.
