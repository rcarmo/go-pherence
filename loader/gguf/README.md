# loader/gguf

GGUF model-file loader: parsing, tensor dequantization, and tensor inspection.

| File | Role |
|---|---|
| `gguf.go` | GGUF container parsing (header, KV metadata, tensor directory) |
| `dequant.go` | Dequantization to `[]float32` (F32/F16/Q8_0/… ); uses `half.F16ToF32` |
| `quant_matrix.go`, `expert_matrix.go` | Quantized weight-matrix accessors (dense + MoE experts) |
| `qdot.go`, `qdot_*.go`, `qdot_*.s` | GGUF quantised dot/GEMV kernels and architecture dispatch, including exact Q4_0 × Q8_0 AVX-VNNI decode kernels |
| `quant_project.go` | Batched quantised projection; long Q4_0 prefill uses parallel Q8_0 activation quantisation plus the exact one-row/eight-token SoA AVX-VNNI tile |
| `tokenizer.go` | GGUF-embedded tokenizer/vocab |
| `inspect.go`, `reap_inspect.go` | Tensor inventory / REAP inspection helpers |

General quantisation codecs live in `backends/simd/quant`; this package handles the GGUF file format and owns format-specific GGUF projection kernels where preserving ggml-compatible block layout and arithmetic order is part of the contract. The retained Gemma4 Q4_0 prefill path is documented in the [CPU SIMD gap note](../../docs/gemma4-cpu-simd-gap.md).
