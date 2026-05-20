# SIMD runtime subpackage move table

Generated/applied by `scripts/split_simd_runtime.py`.

| Source | Target package/path |
|---|---|
| `backends/simd/runtime/activation.go` | `backends/simd/activation/activation.go` |
| `backends/simd/runtime/activation_checked_test.go` | `backends/simd/activation/activation_checked_test.go` |
| `backends/simd/runtime/bias.go` | `backends/simd/activation/bias.go` |
| `backends/simd/runtime/bias_test.go` | `backends/simd/activation/bias_test.go` |
| `backends/simd/runtime/attention.go` | `backends/simd/attention/attention.go` |
| `backends/simd/runtime/attention_test.go` | `backends/simd/attention/attention_test.go` |
| `backends/simd/runtime/simd_amd64.s` | `backends/simd/dot/dot_amd64.s` |
| `backends/simd/runtime/simd_amd64_dispatch.go` | `backends/simd/dot/dispatch_amd64.go` |
| `backends/simd/runtime/simd_arm64.s` | `backends/simd/dot/dot_arm64.s` |
| `backends/simd/runtime/simd_arm64_dispatch.go` | `backends/simd/dot/dispatch_arm64.go` |
| `backends/simd/runtime/simd_other.go` | `backends/simd/dot/dot_other.go` |
| `backends/simd/runtime/scalar.go` | `backends/simd/scalar/dot.go` |
| `backends/simd/runtime/gebp*.{go,s}` | `backends/simd/matmul/gebp*.{go,s}` |
| `backends/simd/runtime/gemv.go` | `backends/simd/matmul/gemv.go` |
| `backends/simd/runtime/gemv_test.go` | `backends/simd/matmul/gemv_test.go` |
| `backends/simd/runtime/pack*.{go,s}` | `backends/simd/matmul/pack*.{go,s}` |
| `backends/simd/runtime/sgemm*.{go,s}` | `backends/simd/matmul/sgemm*.{go,s}` |
| `backends/simd/runtime/layernorm.go` | `backends/simd/norm/layernorm.go` |
| `backends/simd/runtime/layernorm_test.go` | `backends/simd/norm/layernorm_test.go` |
| `backends/simd/runtime/rope.go` | `backends/simd/rope/rope.go` |
| `backends/simd/runtime/rope_freqs.go` | `backends/simd/rope/freqs.go` |
| `backends/simd/runtime/rope_freqs_test.go` | `backends/simd/rope/freqs_test.go` |
| `backends/simd/runtime/rope_test.go` | `backends/simd/rope/rope_test.go` |
| `backends/simd/runtime/softmax.go` | `backends/simd/softmax/softmax.go` |
| `backends/simd/runtime/softmax_test.go` | `backends/simd/softmax/softmax_test.go` |
| `backends/simd/runtime/vec*.{go,s}` | `backends/simd/vector/*.{go,s}` |

Facade-only files kept under `backends/simd/runtime` for now: `bf16*.go`, `capabilities.go`, `checked.go`, and `import_boundary_test.go`.
