# Benchmark snapshot queue

Benchmark entrypoints are present for the current hot primitive matrix. The queue below reflects the 2026-05-20 refreshed phase snapshot recorded in `docs/performance.md`.

Run:

```bash
GOTMPDIR=$PWD/.gotmp go test ./model -run '^$' -bench 'BenchmarkCPUHot' -benchmem
```

## Current benchmark entrypoints

| Benchmark | Primitive / path | Snapshot status |
|---|---|---|
| `BenchmarkCPUHotRMSNorm3584` | SIMD/F32 RMSNorm | Refreshed 2026-05-20. |
| `BenchmarkCPUHotGELUTanhMul8192` | scalar SIMD-owned GELU×Mul | Refreshed 2026-05-20. |
| `BenchmarkCPUHotSiLUMul8192` | scalar SIMD-owned SiLU×Mul | Refreshed 2026-05-20. |
| `BenchmarkCPUHotVecScale3584` | SIMD vector scale | Refreshed 2026-05-20. |
| `BenchmarkCPUHotRoPEPartialGemma4SWA` | Gemma4 SWA RoPEPartial | Refreshed 2026-05-20. |
| `BenchmarkCPUHotRoPEPartialGemma4Full` | Gemma4 full-attention RoPEPartial | Refreshed 2026-05-20. |
| `BenchmarkCPUHotRoPEQwenFull` | Qwen full RoPE | Refreshed 2026-05-20. |
| `BenchmarkCPUHotGQAAttentionDecode512` | CPU GQA attention decode | Refreshed 2026-05-20. |
| `BenchmarkCPUHotGemvMLQ1536x2048` | MLX scalar GEMV | Refreshed 2026-05-20. |
| `BenchmarkCPUHotDequantMLX1536x2048` | caller-owned MLX dequant | Refreshed 2026-05-20. |
| `BenchmarkCPUHotGemmMLXBatch8_1536x2048` | scalar batched MLX GEMV | Refreshed 2026-05-20. |
| `BenchmarkCPUHotMoEMLXExperts512x1024Top2` | synthetic MLX MoE expert fallback | Refreshed 2026-05-20. |
| `BenchmarkCPUHotGemvQ4Sym1536x2048` | Q4/GPTQ scalar GEMV | Refreshed 2026-05-20. |
| `BenchmarkCPUHotGemvQ4Asym1536x2048` | Q4/GPTQ asymmetric scalar GEMV | Refreshed 2026-05-20. |
| `BenchmarkCPUHotMoEQ4Experts512x1024Top2` | synthetic Q4 MoE expert fallback | Refreshed 2026-05-20. |
| `BenchmarkCPUHotDequantQ4Asym1536x2048` | caller-owned asymmetric Q4/GPTQ dequant | Refreshed 2026-05-20. |
| `BenchmarkCPUHotDequantQ4Sym1536x2048` | caller-owned symmetric Q4/GPTQ dequant | Refreshed 2026-05-20. |
| `BenchmarkCPUHotDequantNVFP4_1536x2048` | caller-owned NVFP4 dequant | Refreshed 2026-05-20. |
| `BenchmarkCPUHotGemmNVFP4Batch4_1536x2048` | scalar batched NVFP4 GEMM | Refreshed 2026-05-20. |
| `BenchmarkCPUHotGemvNVFP4_1536x2048` | NVFP4 scalar GEMV | Refreshed 2026-05-20. |

Update `docs/performance.md` and `docs/cpu-simd-coverage.md` after future phase-level benchmark runs.
