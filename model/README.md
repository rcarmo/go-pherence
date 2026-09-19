# model

All model implementation source lives here. The root Go package owns the shared
decoder runtime: loading, forward passes, MoE, RoPE, attention, KV integration and
speculative/multi-token-prediction (MTP) decoding. Family subpackages contain
language, encoder, image, speech and scoring implementations.

| Area | Files |
|---|---|
| LLaMA / GGUF core | `llama.go`, `llama_types.go`, `gguf_llama.go`, `gguf_graph.go`, `gguf_kv_layers.go`, `gguf_qwennext.go` |
| Forward pass | `forward_layer.go`, `attention.go`, `rope.go`, `linear_ops.go`, `batch_prefill.go`, `cpu_decode_step.go`, `chunked_lm_head.go` |
| MoE | `moe.go`, `moe_gpu.go`, `gguf_moe_forward.go`, `reap.go`, `reap_summary.go` |
| Quant | `gguf_quant_rvv.go`, `gguf_quant_cgo.go`, `gguf_turboquant.go`, `bf16.go` |
| Speculative / MTP | `speculative*.go`, `mtp_*.go` |
| GPU | `gpu_forward.go`, `mtp_*_gpu.go` |

## Source and downloaded weights

`model/` is source. BERT, Whisper, speaker (including Community-1) and OmniVoice
now live under this tree alongside the other families. Go imports use
`github.com/rcarmo/go-pherence/model/<family>`; the old `models/<family>` package
paths have no compatibility wrappers.

`checkpoints/` at the repository root is git-ignored data: downloaded weights,
configuration and tokenizer files. `cmd/models/` remains the inspector command
group, and `docs/models/` remains model documentation. Neither is an asset folder.
See the [asset migration notes](../docs/guides/model-assets.md) for older checkouts.

## Layering

Numeric kernels are **not** defined here — RMSNorm/Gemv/RoPE/SiLU are thin wrappers
over `backends/simd`, half-precision conversion is in `half`, and quant codecs are
in `backends/simd/quant` + `backends/spacemit`. This package is orchestration.
