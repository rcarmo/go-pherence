# loader/gguf

GGUF model-file loader: parsing, tensor dequantization, and tensor inspection.

| File | Role |
|---|---|
| `gguf.go` | GGUF container parsing (header, KV metadata, tensor directory) |
| `dequant.go` | Dequantization to `[]float32` (F32/F16/Q8_0/… ); uses `half.F16ToF32` |
| `quant_matrix.go`, `expert_matrix.go` | Quantized weight-matrix accessors (dense + MoE experts) |
| `tokenizer.go` | GGUF-embedded tokenizer/vocab |
| `inspect.go`, `reap_inspect.go` | Tensor inventory / REAP inspection helpers |

Quantization codecs themselves live in `backends/simd/quant`; this package handles
the GGUF file format and routes to those codecs.
