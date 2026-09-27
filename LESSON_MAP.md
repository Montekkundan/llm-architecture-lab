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
| 36 | MLA latent cache | `preset("pico-mla")`; `attention/mla.py`; `tests/test_model.py` | Unabsorbed decoder, latent-cache accounting and CPU shape/backward/cache parity implemented |
| 37 | Decoupled RoPE and absorbed MLA weights | Decoupled RoPE in `attention/mla.py`; add absorbed/unabsorbed numerical parity and cost study | Positional path implemented; absorption next |
| 38 | Sliding/global attention | Add causal window masks, mixed-layer schedule and exact-mask tests | Next exercise |
| 39 | Block-sparse attention | Add selection operator and dense-reference/recall tests | Next exercise |
| 40 | Linear/recurrent attention | Add recurrent state and step/sequence parity tests | Next exercise |
| 41 | Norm placement and QK norm | Extend `DecoderBlock` with validated modes and gradient ablation | Next exercise |
| 42 | Gates and residual scaling | Add controlled equal-budget stability experiment | Next exercise |
| 43 | GELU versus SwiGLU versus expert FFN | Add matched-parameter variants and ablation | SwiGLU implemented; comparison next |
| 44 | MoE routing | `preset("pico-moe")`; `moe/router.py`; `tests/test_model.py` | Top-k decoder and CPU dispatch, weight conservation and gradients implemented; capacity/distributed routing next |
| 45 | Shared experts and balancing | Add utilization metrics and controlled routing ablation | Next exercise |
| 46 | Multi-token prediction | Add training head, weighted loss and inference test | Next exercise |
| 47 | Cache across attention families | `model.py:ModelCache` and cached/full parity for MHA/MQA/GQA/YaRN/MLA/MoE | Selected CPU paths implemented; other families next |
| 48 | IO-aware exact attention | Add online-softmax block reference, GPU kernel and profiler protocol | Next exercise |
| 49 | Numerical precision | Add BF16/FP8 environment-gated tests and error report | FP32 checks only; rest next |
| 50 | Parameter, FLOP, cache and communication costs | `spec.py:kv_cache_bytes`; add measured parameter/FLOP/mesh estimator | KV payload only; rest next |

The model tests are `tests/test_model.py` and require PyTorch. The
dependency-free spec tests are `tests/test_spec.py`. Running all tests is the
completion gate for current code; it is not evidence for unfinished targets.
