# CPU SIMD Coverage Audit

This is the baseline for Phase 4b: making CPU inference hit AVX2/FMA or NEON
wrappers for every hot decode/prefill primitive, with scalar Go as fallback.
The SIMD implementation now lives at import path `github.com/rcarmo/go-pherence/backends/simd/runtime` (package name `simd`).

## Current coverage map

| Hot path | Current implementation | AVX2/FMA | NEON | Notes / next action |
|---|---|---:|---:|---|
| `RMSNorm` F32 | `simd.RMSNorm` wrapper | ✅ | ✅ | Used by decoder and `ForwardLayer` |
| `RMSNormBF16` on F32 buffers | `simd.RMSNormBF16` wrapper | ✅ | ✅* | arm64 runtime verification pending |
| `RMSNormNoScale` | `simd.RMSNormNoScale` wrapper | ✅ | ✅* | Gemma4 CPU V norm now routes through wrapper |
| Residual add | `simd.VecAdd` | ✅ | ✅ | Decoder and `ForwardLayer` use wrapper |
| Residual + scale | `simd.VecScaleAdd` / `simd.VecScale` | ✅ | ✅ | Gemma4 layer scalar now uses `VecScale` |
| `ToBF16` | `simd.ToBF16` | ✅ | ✅ | Used for Gemma3/4 truncation semantics |
| SiLU × Mul | `simd.VecSiLUMul` wrapper → `backends/simd/kernels.SiLUMul` scalar kernel | wrapper only | wrapper only | Ownership moved to SIMD kernels; future SIMD approximation tolerance: max abs `1e-4` vs scalar |
| GELU(tanh) × Mul | `simd.GELUTanhMul` wrapper → `backends/simd/kernels.GELUTanhMul` scalar kernel | wrapper only | wrapper only | Ownership moved to SIMD kernels; future SIMD approximation tolerance: max abs `1e-4` vs scalar |
| RoPE | `backends/simd/runtime.ApplyRoPE` explicit dispatch hook → scalar kernel | ❌ | ❌ | Ownership moved to SIMD; `HasRoPE=false` until vectorized pair rotation lands |
| RoPEPartial | `backends/simd/runtime.ApplyRoPEPartial` explicit dispatch hook → scalar kernel | ❌ | ❌ | Ownership moved to SIMD; high priority for Gemma4 CPU path |
| GQA attention scores | `simd.Sdot` per head/token | ✅ | ✅ | Intermediate improvement; still allocates scores per head |
| GQA attention output | `simd.Saxpy` per cached-token V head | ✅ | ✅ | Caller-owned output/score scratch; full fused attention still future work |
| F32 GEMV dense | checked `simd.SgemmNNTo` / `simd.GemvRows` / `simd.GemvCols` references | ✅ | ✅ | model/tensor/BERT callers use checked runtime entrypoints; unsafe SGEMM calls are boundary-guarded |
| MLX4 GEMV | `backends/mlx` scalar unpack/dequant loop with dtype/shape validation; model/backend code imports it directly | ❌ | ❌ | Explicit `HasGemv4=false`/`HasDequant=false` capability gates, scalar dispatch hook, caller-owned dequant output helper, and scalar batched `Gemm` API; biggest CPU gap for quantized models and MoE experts |
| GPTQ Q4 GEMV | `backends/simd/quant/q4` scalar unpack/dequant loop with qweight/g_idx/scales/qzeros validation; model/backend code imports it directly | ❌ | ❌ | Scalar symmetric GEMV now traverses packed rows contiguously, dequant supports caller-owned output, and explicit `HasGemvSym=false`/`HasDequant=false` gates are present; still needs AVX2/NEON nibble unpack + FMA |
| NVFP4 GEMV/dequant | `backends/simd/quant/nvfp4` correctness-first scalar decode/dequant/GEMV | ❌ | ❌ | Explicit `HasDecode=false`/`HasDequant=false`/`HasGemv=false` gates, caller-owned dequant output helper, and `GemvNVFP4Reference` scalar parity target |
| MoE CPU experts | parallel goroutines + MLX4 scalar GEMV | partial | partial | Activation now goes through SIMD wrapper; GEMV dominates |
| BERT/GTE encoder | workspace + SGEMM/SIMD vec ops | ✅ | ✅ | Already comparatively mature |
| TurboQuant rotation/dequant | scalar matvec + bit unpack | ❌ | ❌ | Needs scratch reuse and SIMD matvec/unpack |

`✅*` means the assembly exists/cross-compiles but still needs runtime proof on arm64 hardware.

## Benchmarks added

`model/cpu_hotpath_bench_test.go` adds synthetic baselines for:

- `BenchmarkCPUHotRMSNorm3584`
- `BenchmarkCPUHotGELUTanhMul8192`
- `BenchmarkCPUHotSiLUMul8192`
- `BenchmarkCPUHotVecScale3584`
- `BenchmarkCPUHotRoPEPartialGemma4SWA`
- `BenchmarkCPUHotRoPEPartialGemma4Full`
- `BenchmarkCPUHotRoPEQwenFull`
- `BenchmarkCPUHotGQAAttentionDecode512`
- `BenchmarkCPUHotGemvMLX1536x2048`
- `BenchmarkCPUHotDequantMLX1536x2048`
- `BenchmarkCPUHotGemmMLXBatch8_1536x2048`
- `BenchmarkCPUHotMoEMLXExperts512x1024Top2`
- `BenchmarkCPUHotGemvQ4Sym1536x2048`
- `BenchmarkCPUHotGemvQ4Asym1536x2048`
- `BenchmarkCPUHotMoEQ4Experts512x1024Top2`
- `BenchmarkCPUHotDequantQ4Asym1536x2048`
- `BenchmarkCPUHotDequantQ4Sym1536x2048`
- `BenchmarkCPUHotDequantNVFP4_1536x2048`
- `BenchmarkCPUHotGemmNVFP4Batch4_1536x2048`
- `BenchmarkCPUHotGemvNVFP4_1536x2048`

Run with (see [benchmark-snapshot-queue.md](benchmark-snapshot-queue.md) for current snapshot status):

```bash
GOTMPDIR=$PWD/.gotmp go test ./model -run '^$' -bench 'BenchmarkCPUHot' -benchmem
```

## Dispatch cleanup status

- `RuntimeCapabilities()` in the `backends/simd/runtime` package centralizes architecture/runtime feature reporting.
- `simd.HasSgemmAsm`, `simd.HasDotAsm`, and `simd.HasVecAsm` expose runtime-safe capability gates.
- `Sdot`/`Saxpy` now dispatch through small Go wrappers and fall back to scalar code if AVX2/FMA or NEON is unavailable, or if callers pass mismatched lengths.
- Non-SIMD SGEMM callers use checked slice APIs (`SgemmNNTo`/`SgemmNTTo`) that validate dimensions, leading dimensions, backing slices, and stride/product overflow before invoking unsafe assembly or scalar fallback. Boundary tests prevent direct unsafe SGEMM calls outside the SIMD runtime. `SgemmNTGebp` and `SgemmNTBlockedFMA` also validate dimensions, pointers, strides, and overflow before unsafe slicing/pointer arithmetic.
- Vector entrypoints (`VecAdd`, `VecMul`, `VecScaleAdd`, `RMSNorm*`, `ToBF16`, BF16 helpers) now dispatch through Go wrappers and fall back to scalar code when runtime SIMD gates are false. Scalar fallbacks bound all participating slices and leave untouched destination tails unchanged on malformed inputs.
- Activation wrappers (`VecSiLUMul`, `GELUTanhMul`) now route through scalar kernels in `backends/simd/kernels/activation.go`; `RuntimeCapabilities().HasActivation` remains false until AVX2/NEON polynomial approximations land and pass the explicit tolerance gate.
- RoPE wrappers now have explicit runtime dispatch hooks and a `RuntimeCapabilities().HasRoPE` flag. It remains false until AVX2/NEON kernels land and pass scalar parity tests.

## Baseline snapshot (i7-12700, amd64)

```text
BenchmarkCPUHotRMSNorm3584                  649 ns/op, 0 allocs
BenchmarkCPUHotGELUTanhMul8192              101 µs/op, 0 allocs
BenchmarkCPUHotSiLUMul8192                  82.7 µs/op, 0 allocs
BenchmarkCPUHotVecScale3584                 154 ns/op, 0 allocs
BenchmarkCPUHotRoPEPartialGemma4SWA         3.27 µs/op, 0 allocs
BenchmarkCPUHotRoPEPartialGemma4Full        1.74 µs/op, 0 allocs
BenchmarkCPUHotRoPEQwenFull                 6.36 µs/op, 0 allocs
BenchmarkCPUHotGQAAttentionDecode512        271 µs/op, 0 allocs
BenchmarkCPUHotGemvMLQ1536x2048             3.74 ms/op, 0 allocs
BenchmarkCPUHotGemmMLXBatch8_1536x2048      5.08 ms/op, 9 allocs
BenchmarkCPUHotDequantMLX1536x2048          1.73 ms/op, 13 allocs
BenchmarkCPUHotMoEMLXExperts512x1024Top2    1.60 ms/op, 13 allocs
BenchmarkCPUHotGemvQ4Sym1536x2048           4.43 ms/op, 0 allocs
BenchmarkCPUHotGemvQ4Asym1536x2048          8.15 ms/op, 0 allocs
BenchmarkCPUHotMoEQ4Experts512x1024Top2     915 µs/op, 0 allocs
BenchmarkCPUHotDequantQ4Asym1536x2048       1.79 ms/op, 13 allocs
BenchmarkCPUHotDequantQ4Sym1536x2048        3.77 ms/op, 13 allocs
BenchmarkCPUHotDequantNVFP4_1536x2048       1.25 ms/op, 13 allocs
BenchmarkCPUHotGemmNVFP4Batch4_1536x2048    7.11 ms/op, 10 allocs
BenchmarkCPUHotGemvNVFP4_1536x2048          11.0 ms/op, 0 allocs
```

## Immediate next steps

1. Vectorize `RoPEPartial` on AVX2 and NEON.
2. Add MLX4 GEMV SIMD kernels for CPU quantized decode and MoE experts.
3. Extend caller-owned scratch-buffer reuse to MLP, PLI, MoE, and TurboQuant decode paths.
4. Add allocation gates for decode once broader scratch-buffer reuse lands.
5. Runtime-verify arm64 NEON kernels on Orange Pi 6+.


## Folder reorg note

The current `backends/simd/runtime` package keeps architecture-specific files in one package with Go build tags (`*_amd64.go`, `*_arm64.go`, `*_other.go`). Phase 6.6 has started with facade-preserving cleanup: scalar dot/SAXPY fallbacks are now in `scalar.go`, scalar RMSNorm uses precise `math.Sqrt`, BF16 GEMV validates shape-product overflow, and SGEMM/GEBP/gather wrappers validate capability gates and overflow-prone pointer arithmetic. A literal subfolder split would create separate Go packages, so keep `backends/simd/runtime` as the public facade and split internals only after wrapper boundaries are explicit. See `docs/simd-folder-reorg.md`.
