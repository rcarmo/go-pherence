# Benchmark snapshot queue

Benchmark entrypoints are present for the current hot primitive matrix, but several numeric snapshots should be refreshed together during the next phase validation run rather than after each small change.

Run:

```bash
GOTMPDIR=$PWD/.gotmp go test ./model -run '^$' -bench 'BenchmarkCPUHot' -benchmem
```

## Current benchmark entrypoints

| Benchmark | Primitive / path | Snapshot status |
|---|---|---|
| `BenchmarkCPUHotRMSNorm3584` | SIMD/F32 RMSNorm | Existing snapshot in `docs/cpu-simd-coverage.md`. |
| `BenchmarkCPUHotGELUTanhMul8192` | scalar SIMD-owned GELU×Mul | Existing snapshot after Phase 2 tolerance pass. |
| `BenchmarkCPUHotSiLUMul8192` | scalar SIMD-owned SiLU×Mul | Existing snapshot after Phase 2 tolerance pass. |
| `BenchmarkCPUHotVecScale3584` | SIMD vector scale | Existing snapshot. |
| `BenchmarkCPUHotRoPEPartialGemma4SWA` | Gemma4 SWA RoPEPartial | Existing snapshot after Phase 1 pass. |
| `BenchmarkCPUHotRoPEPartialGemma4Full` | Gemma4 full-attention RoPEPartial | Existing snapshot after Phase 1 pass. |
| `BenchmarkCPUHotRoPEQwenFull` | Qwen full RoPE | Existing snapshot after Phase 1 pass. |
| `BenchmarkCPUHotGQAAttentionDecode512` | CPU GQA attention decode | Existing snapshot. |
| `BenchmarkCPUHotGemvMLQ1536x2048` | MLX scalar GEMV | Existing snapshot. |
| `BenchmarkCPUHotDequantMLX1536x2048` | caller-owned MLX dequant | Pending Phase 4/10 snapshot. |
| `BenchmarkCPUHotGemmMLXBatch8_1536x2048` | scalar batched MLX GEMV | Pending Phase 4/10 snapshot. |
| `BenchmarkCPUHotMoEMLXExperts512x1024Top2` | synthetic MLX MoE expert fallback | Pending Phase 4/10 snapshot. |
| `BenchmarkCPUHotGemvQ4Sym1536x2048` | Q4/GPTQ scalar GEMV | Pending refresh after scalar traversal change. |
| `BenchmarkCPUHotDequantQ4Sym1536x2048` | caller-owned Q4/GPTQ dequant | Pending Phase 3/10 snapshot. |
| `BenchmarkCPUHotDequantNVFP4_1536x2048` | caller-owned NVFP4 dequant | Pending Phase 5/10 snapshot. |
| `BenchmarkCPUHotGemvNVFP4_1536x2048` | NVFP4 scalar GEMV | Existing snapshot, refresh after Phase 5 additions. |

Update `docs/performance.md` and `docs/cpu-simd-coverage.md` after the phase-level benchmark run.
