# SIMD inference matmul policy

The CPU inference path separates autoregressive decode from prompt and verifier prefill because they have different useful kernels.

## Shape-aware dispatch

`GemmRowsParallel` handles row-major weights stored as `[output,input]`:

- `batch == 1` retains the low-latency SIMD dot/GEMV path. Decode is memory-bandwidth dominated and gains nothing from cache blocking a single activation row.
- `2 <= batch <= 256`, with both matrix dimensions at least 64, uses the cache-blocked NT kernel. Its 2-output microkernel shares each activation load across two weight rows, while the driver blocks K and output rows in groups of 64.
- Larger batches retain the established parallel dot path. Whisper encoder matrices commonly have about 1,500 rows; on the current i7-12700 the blocked path regressed the complete MOSS CPU run, so it is deliberately excluded until a microkernel also tiles the activation-row dimension.
- Workers own disjoint 64-row output tiles. This mirrors llamafile's caller-aware tile partitioning and avoids both output contention and nested per-element scheduling.

The policy follows the useful part of Justine Tunney's [LLaMA Now Goes Faster on CPUs](https://justine.lol/matmul/): multi-output register tiles improve matrix-matrix prompt work by reusing loaded operands, while matrix-vector token generation needs a separate low-overhead kernel. It does not copy llamafile's full kernel matrix; go-pherence currently has AVX2/FMA and NEON assembly, so dispatch remains bounded to shapes measured here.

## Reproducing the measurements

```bash
env GOMAXPROCS=1 go test ./backends/simd/runtime \
  -run '^$' -bench '^BenchmarkArticleNT' -benchtime=500ms -count=3

env GOMAXPROCS=2 go test ./backends/simd/runtime \
  -run '^$' -bench '^BenchmarkArticleGemmRows' -benchtime=500ms -count=3
```

On the i7-12700 development host, the serial blocked kernel is roughly 1.2-1.5x faster than the one-output NT kernel at the 32-row and 227-row prefill shapes, while `batch=1` remains faster on the original dot kernel. End-to-end MOSS transcript parity remains exact; benchmark variance under normal host load is large enough that kernel microbenchmarks and the complete JFK fixture are both required before widening the dispatch range.
