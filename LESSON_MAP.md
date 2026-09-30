# Lessons 31–50: code and check map

`Implemented` means the current starter includes an operator and a test for
the claim. `Next exercise` means a lesson target, **not** a delivered feature.

| Lesson | Topic | File or command | Status |
| --- | --- | --- | --- |
| 31 | Versioned architecture specification | `src/architecture_lab/spec.py`; `tests/test_spec.py` | Implemented |
| 32 | Tiny versus published-size model | `preset("llama3-8b-tiny")`, `preset("llama3-8b-reference")`; `tests/test_spec.py` | Implemented metadata split; not a full reproduction |
| 33 | RoPE identities | `model.py:apply_rope`; `tests/test_model.py::test_rope_preserves_norm_and_relative_dot_products` | Implemented; PyTorch check pending in environments without torch |
| 34 | YaRN context extension | `preset("pico-yarn")`; `position/yarn.py`; `tests/test_model.py` | Fixed-scale decoder/cache and CPU identities implemented; long-context evaluation pending |
| 35 | MHA, MQA and GQA cache contracts | `spec.py:attention_kind`, `model.py:CausalAttention`; `python -m architecture_lab.checks --preset pico-gqa` | Implemented; PyTorch checks pending where torch absent |
| 36 | MLA latent cache | `preset("pico-mla")`; `attention/mla.py`; `tests/test_model.py` | Reconstruction decoder, latent-cache accounting and CPU shape/backward/cache parity implemented |
| 37 | Decoupled RoPE and absorbed MLA weights | `preset("pico-mla-absorbed")`; `attention/mla.py`; `tests/test_mechanisms.py` | Absorbed cached inference and double-precision parity implemented; latency/cost study next |
| 38 | Sliding/global attention | `labs.py:causal_window_mask`; `tests/test_labs.py` | Absolute-position local/global mask and dense oracle implemented; hybrid decoder/cache eviction and quality lab remain separate |
| 39 | Block-sparse attention | `labs.py:selected_block_mask`, `masked_attention`; `tests/test_labs.py` | Deterministic current-block selection, causality and dense-mask parity implemented; learned indexer/sparse kernel not claimed |
| 40 | Linear/recurrent attention | `labs.py:LinearState`, `linear_attention`; `tests/test_labs.py` | ELU+1 normalized kernel recurrence and chunked output/gradient parity implemented; not KDA or DeltaNet |
| 41 | Norm placement and QK norm | `labs.py:qk_normalize`, `residual_branch`; `tests/test_labs.py` | Feature-axis QK norm and four explicit residual equations implemented; model-scale gradient ablation next |
| 42 | Gates and residual scaling | `labs.py:residual_branch`; `tests/test_labs.py` | Token gate/fixed or learned scale, zero-scale gradient experiment implemented; trained equal-budget comparison next |
| 43 | GELU versus SwiGLU versus expert FFN | `labs.py:GeluFFN`, `model.py:SwiGLU`; `tests/test_labs.py` | Matched bias-free parameter count and module paths implemented; trained ablation next |
| 44 | MoE routing | `preset("pico-moe")`, `preset("pico-mla-yarn-moe")`; `moe/router.py`; `tests/test_model.py` | Top-k decoder, combined MLA/YaRN/MoE path, CPU dispatch, weight conservation and gradients implemented; capacity/distributed routing next |
| 45 | Shared experts and balancing | `moe/router.py:TopKMoE`; `tests/test_labs.py` | Optional shared path, unbiased gate weights, auxiliary balance term and selection-only bias update implemented; no distributed router recipe |
| 46 | Multi-token prediction | `labs.py:MultiTokenHeads`, `multitoken_loss`; `tests/test_labs.py` | Independent horizon heads with document-boundary masks and weighted loss implemented; not DeepSeek sequential MTP or speculative sampler |
| 47 | Cache across attention families | `model.py:ModelCache` and cached/full parity for MHA/MQA/GQA/YaRN/MLA/MoE | Selected CPU paths implemented; other families next |
| 48 | IO-aware exact attention | `labs.py:online_attention`; `tests/test_labs.py` | Online-softmax tile reference with masked output/backward parity implemented; fused GPU kernel/profiling not claimed |
| 49 | Numerical precision | `checks.py --labs`; `tests/test_labs.py` | Seeded BF16/FP32 logits error and FP32 reduction tests implemented; FP8/GPU throughput/held-out evaluation not claimed |
| 50 | Parameter, FLOP, cache and communication costs | `labs.py:parameter_count`, `training_matmul_flops`, `ring_allreduce_bytes`; `spec.py:kv_cache_bytes` | Exact decoder parameter count and explicit 6ND/KV/ring transfer estimates implemented; measured GPU/mesh costs not claimed |

The model tests are `tests/test_model.py` and require PyTorch. The
dependency-free spec tests are `tests/test_spec.py`. Running all tests is the
completion gate for current code; it is not evidence for unfinished targets.
