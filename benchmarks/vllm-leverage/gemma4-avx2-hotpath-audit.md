# Gemma4 E4B CPU/AVX2 hot-path audit

Date: 2026-08-07. Canonical E4B QAT Q4_0 GGUF (`4b4a2c1d`, SHA-256 `676c3507`). Loaded dimensions: hidden 2560, intermediate 10240, 42 layers, 8 query heads, 2 KV heads, head width 256, PLI width 256, vocabulary 262144.

| Operation | Decode/batch shape | Current path | Gap / disposition |
|---|---|---|---|
| Input/final/Q/K RMSNorm | 2560 or 256; M=1 | AVX2 vector dispatch through `simd.RMSNorm`; no-scale V norm also dispatched | Covered; retain scalar fallback and add dispatch counters before further tuning. |
| Q/K/V projections | 2560→2048 / 512 / 512 | GGUF quantized projection; verifier batch calls `ProjectBatchF32To` for M>1 | First Tunney target: preserve M=1 quant GEMV reduction order; share Q4_0 weight decode across M=2/4/8 rows only behind exact oracle. |
| O projection | 2048→2560 | Same quant projection families | Same M=1/M>1 split; measure separately because output-row count differs. |
| Gate/up/down | 2560→10240→2560 | Quantized projection plus SIMD GELU-multiply; existing backend batch x4 paths cover several GGUF formats | Highest byte-volume candidate. Benchmark fused gate/up traversal and four-row blocking; never route M=1 through a slower GEMM. |
| PLI projection/gating | model projection 2560→42×256; per-layer gate 2560→256 and projection 256→2560 | GGUF/dense projections plus GELU-multiply, post RMSNorm and residual | Shape-specialize the 256-wide gate/input and 256→2560 projection; avoid materializing per-request pointer slices in batch mode. |
| RoPE | 8×256 Q, 2×256 K | Existing in-place path; not a dominant matrix multiply | Keep vectorized math/tables; no Tunney matmul work. |
| GQA attention | heads 8/2, head 256, sequence dependent | SIMD dot/softmax/SAXPY and parallel-head threshold | Covered for long contexts; static batch needs flattened lengths/masks before cross-request SIMD is safe. |
| Residual/GELU/scale | 2560 or 10240 | SIMD vector kernels | Covered; fuse only with measured allocation/traffic benefit. |
| LM head | 2560→262144 | M=1 `GemvRowsParallel`; M>1 `DenseNTTo` in `FinishCPUDecodeBatch` | Strong Tunney target. Existing E4B batch-tail measurements show B2/B4/B8 same-batch gains but B1 regression, so dispatch must keep M=1 GEMV. |
| KV append/copy | 2×256 per owning layer/token | Go slice append/copy; attention reads contiguous float KV | Instrument bytes and allocation growth first; logical paging is not justified by this audit alone. |
| Sampling | 262144 logits | argmax/softcap/suppression | Greedy session path is exact; profile separately from matrix kernels. |

## Tunney/llama.cpp specialization rule

Treat matrix row count as a first-class dispatch key rather than forcing one generic GEMM:

- **M=1 decode:** retain output-row-unrolled GEMV, quantize/decode the activation once, process multiple weight rows per loop where the quant format preserves the established accumulation order, and keep alignment/tail scalar cleanup.
- **M=2/4/8:** tile activation rows together so each packed weight block and scale is loaded/decoded once and accumulated into 2/4/8 independent vectors. Prefer the existing x4 backend kernels and compose tails as 4+2+1.
- Dispatch additionally by `(inDim,outDim)` for Q/K/V, O, gate/up/down, PLI and LM head; a win at 2560×262144 must not silently select a losing kernel at 2560×512.
- Promotion requires scalar/repeated-GEMV parity, amd64 feature/fallback tests, arm64/riscv64 builds, and full-session TTFT/ITL or aggregate-batch improvement. Microkernel wins alone are insufficient.

## Immediate measured implication

The real E4B static-tail sweep (`c6c98914`) confirms the dispatch boundary: batched final norm/LM-head improves B2/B4/B8 versus repeated same-batch execution, but B1 is 7.6% slower. Therefore M=1 must remain on the decode GEMV route while M>1 specialization is evaluated per shape.

## First retained shape dispatch

The first Q4_0 candidate reuses Q8_0 activation quantization and one row-parallel traversal at **B8 only**. Synthetic E4B dimensions on the i7-12700 (`-benchtime=10x -count=3`) reduced allocations from 128–133 to 24–25 per projection and improved the median versus eight repeated GEMVs at output widths 512 (2.69ms vs 3.76ms), 2048 (10.93ms vs 20.28ms), 2560 (11.91ms vs 13.72ms), and 10240 (63.20ms vs 100.59ms). Variance was material, so B1/B2/B4 deliberately return the existing unsupported-batch signal and retain their established fallback rather than promoting noisy or regressing paths.
