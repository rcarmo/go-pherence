# Backend parity matrix

This document tracks CPU/reference parity targets for backend wrappers. It complements the benchmark snapshots in `docs/performance.md` and the ownership table in `docs/kernel-coverage.md`.

## Reference owners

| Primitive / format | Scalar/reference owner | Runtime/backend paths | Current parity status |
|---|---|---|---|
| RMSNorm | `backends/simd/runtime` scalar fallback and `backends/simd/kernels` helpers | SIMD asm wrappers, NVIDIA RMSNorm, Vulkan wrapper stub | SIMD scalar/asm coverage exists; NVIDIA/Vulkan parity should be hardware-gated. |
| RMSNormNoScale | `backends/simd/runtime.RMSNormNoScale` | NVIDIA RMSNormNoScale, Vulkan wrapper stub | CPU scalar path exists; NVIDIA/Vulkan parity should be hardware-gated. |
| RoPE / RoPEPartial | `backends/simd/kernels.ApplyRoPE*` | SIMD runtime wrapper, NVIDIA RoPE kernels, Vulkan RoPEPartial wrapper stub | Scalar odd/tail/short-freq tests exist; AVX2/NEON and GPU parity pending. |
| SiLU×Mul | `backends/simd/kernels.SiLUMul` | SIMD runtime scalar wrapper, NVIDIA fused SiLU, Vulkan wrapper stub | Scalar golden/tail tests exist; optimized approximation tolerance is `1e-4`; GPU parity pending. |
| GELU(tanh)×Mul | `backends/simd/kernels.GELUTanhMul` | SIMD runtime scalar wrapper, NVIDIA fused GELU, Vulkan wrapper stub | Scalar golden/tail tests exist; optimized approximation tolerance is `1e-4`; GPU parity pending. |
| GQA attention scores/output | `backends/simd/kernels` and model CPU references | NVIDIA attention score/fused attention, Vulkan attention-score wrapper stub | CPU path benchmarked; GPU parity should be availability-gated. |
| MLX4 GEMV/dequant | `backends/mlx` | NVIDIA MLX GEMV/GEMM, future AVX2/NEON, scalar Vulkan N/A | Known-value and caller-owned dequant tests exist; NVIDIA parity exists in runtime tests, future SIMD parity pending. |
| Q4/GPTQ GEMV/dequant | `backends/simd/runtime/q4` | NVIDIA symmetric Q4 GEMV/GEMM, future AVX2/NEON | Scalar-vs-dequant parity test exists; asymmetric qzeros remains CPU/reference only for NVIDIA. |
| NVFP4 decode/dequant/GEMV | `backends/simd/runtime/nvfp4` including `GemvNVFP4Reference` | NVIDIA dense F32 fallback, future packed/native kernels | Synthetic scalar tests and explicit reference target exist; real checkpoint/hardware smoke pending. |
| BF16 runtime helpers | `backends/simd/runtime/bf16` where CPU helper exists | NVIDIA emulated/native BF16 wrappers | See `docs/bf16-parity.md`; BF16 no-scale is NVIDIA-only until a CPU BF16 model path needs it. |

## Test policy

- CPU-only scalar/reference tests should live near the owning package.
- Optimized SIMD tests should compare against the scalar owner and use operation-specific tolerances.
- NVIDIA and Vulkan tests should be availability-gated and skip cleanly when hardware/runtime support is absent.
- Vulkan wrapper boundary tests may run without a Vulkan device; numeric parity waits for pipeline cache wiring.
- Public generation behavior is validated by the full phase gate, not by per-change test runs. See [validation-gates.md](validation-gates.md) for the standard phase-level commands.
- Malformed-input coverage is indexed in [malformed-input-coverage.md](malformed-input-coverage.md).
