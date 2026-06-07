# model

The core **GGUF / LLaMA-family transformer runtime**: model loading, the forward
pass, MoE, RoPE, attention, KV cache, and speculative / multi-token-prediction
(MTP) decoding. Image, audio, and other families live in sibling sub-packages.

| Area | Files |
|---|---|
| LLaMA / GGUF core | `llama.go`, `llama_types.go`, `gguf_llama.go`, `gguf_graph.go`, `gguf_kv_layers.go`, `gguf_qwennext.go` |
| Forward pass | `forward_layer.go`, `attention.go`, `rope.go`, `linear_ops.go`, `batch_prefill.go`, `cpu_decode_step.go`, `chunked_lm_head.go` |
| MoE | `moe.go`, `moe_gpu.go`, `gguf_moe_forward.go`, `reap.go`, `reap_summary.go` |
| Quant | `gguf_quant_rvv.go`, `gguf_quant_cgo.go`, `gguf_turboquant.go`, `bf16.go` |
| Speculative / MTP | `speculative*.go`, `mtp_*.go` |
| GPU | `gpu_forward.go`, `mtp_*_gpu.go` |

## `model/` vs `models/`

- **`model/`** (this package, `package model`) — the LLaMA/GGUF transformer core,
  with sub-packages for specific families: `llama/`, `qwen/`, `gemma/`, `gemma4/`,
  `lfm2/`, `qwen3tts/`, and the image/3D models `ideogram4/`, `hunyuan3d/`,
  `trellis2/`.
- **`models/`** (`package whisper`, `speaker`, `bert`) — standalone non-GGUF model
  implementations with their own pipelines.

## Layering

Numeric kernels are **not** defined here — RMSNorm/Gemv/RoPE/SiLU are thin wrappers
over `backends/simd`, half-precision conversion is in `half`, and quant codecs are
in `backends/simd/quant` + `backends/spacemit`. This package is orchestration.
