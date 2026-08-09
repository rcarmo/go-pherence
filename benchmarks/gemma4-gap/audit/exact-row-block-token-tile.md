# Exact row-block/token-tile projection

## Status

Accepted and promoted after exact correctness and clean packing-inclusive timing. The OpenTranscribe stack was stopped with explicit approval before measurement; Jupyter remained untouched.

## Mechanism

The retained long-prefill loop is row-major: each row consumes all fifteen eight-token activation tiles. A real 80-block tile is 23,040 bytes, while all fifteen tiles occupy 345,600 bytes. That fits a P-core L2 but not L1, so the kernel repeatedly pulls activation vectors from L2 for every output row.

The candidate claims 64-row blocks dynamically and interchanges the inner loops:

```text
for token tile
    for row in claimed 64-row block
        retained exact one-row/eight-token assembly
```

At the real shape, one activation tile is 22.5 KiB and remains within L1 while its claimed rows are processed. Sixty-four Q4 rows occupy 92,160 bytes and remain within L2 while token tiles advance. The assembly, transient Q8 layout, eight FP32 accumulators and final reduction are unchanged.

The worker policy uses five goroutines plus the caller at `GOMAXPROCS=6`. An atomic 64-row claim replaces six static shards, also attacking the measured static-tail mechanism without creating a persistent worker pool.

## Idealised cache-traffic bound

For one 64-row block and the fifteen full eight-token tiles:

| schedule | repeated activation source | repeated Q4 source | idealised L2-to-L1 source |
|---|---:|---:|---:|
| retained row-major | 64 x 345,600 B | 64 x 1,440 B once resident | about 22.21 MB |
| candidate blocked | 345,600 B once | 64 x 1,440 B x 15 tiles | about 1.73 MB |

This is an idealised 92.2% reduction in source bytes crossing from L2 into L1 for the blocked region, not a hardware-counter claim. `perf_event_paranoid=4` prevents direct cache-counter validation. The trade is deliberate: Q4 rows move from L1 reuse to L2 reuse, while the much larger repeated Q8 stream moves from L2 to L1.

No model weights are repacked or duplicated. Allocation and activation packing are unchanged.

## Correctness

The projection tests compare blocked output with scalar/reference output across supported Q4 shapes and tails, including a 513-row/124-token case that exercises parallel claims. A dedicated coverage test proves rows 1, 64, 65, 513 and 10,240 are each visited exactly once, propagates worker failure and rejects malformed ranges. Results:

```text
GOMAXPROCS=6 go test ./loader/gguf \
  -run '^(TestQuantMatrixProjectBatchQ4_0.*|TestDotQ4_0Q8_0Tokens8SoARandomExact)$' \
  -count=20
ok

GOMAXPROCS=6 go test -race ./loader/gguf \
  -run '^(TestGemvRowBlocksParallelCoverage|TestQuantMatrixProjectBatchF32ToMatchesDequantOracle)$' \
  -count=5
ok

go vet ./loader/gguf
ok
```

Each output is still computed by exactly one unchanged assembly call, so there is no new FP reassociation or shared accumulation state.

## Performance result

After a stable 20-second inspection at 93.3--99.3% idle, five alternating baseline/candidate samples used:

```sh
taskset -c 0-5 env GOMAXPROCS=6 go test ./loader/gguf \
  -run '^$' \
  -bench '^BenchmarkQuantMatrixProjectBatchQ4_0Gemma4Shapes/out10240/batch124/batched$' \
  -benchmem -benchtime=1s -count=1
```

Baseline is detached worktree `/tmp/go-pherence-correction-baseline` at `1c1ff55d`. The candidate won all five pairs. Baseline samples were 13.270, 13.277, 12.442, 11.506 and 10.822 ms/op; candidate samples were 10.095, 10.354, 9.950, 9.783 and 10.042 ms/op. Medians were **12.442 versus 10.042 ms/op**, a **19.3% packing-inclusive projection-time reduction**. Allocation also fell from 32--33 to 26--27 allocations/op. The monitor found no non-benchmark process above 20% during any sample; the only later Jupyter health-check process occurred after the final benchmark. Raw timing and load evidence are [`go_q4_row_block_token_tile_projection_bench_clean.log`](go_q4_row_block_token_tile_projection_bench_clean.log) and [`go_q4_row_block_token_tile_projection_load_clean.log`](go_q4_row_block_token_tile_projection_load_clean.log).

The earlier rejected load gate remains in [`go_q4_row_block_token_tile_projection_bench.log`](go_q4_row_block_token_tile_projection_bench.log). Subsequent attribution showed that the substantial recurring interference came from four OpenTranscribe Celery worker checks, each consuming roughly 2.4--2.7 seconds every 30 seconds. Overlap drove the six-CPU host as high as 53.2% user plus 5.0% system CPU. Other checks were brief; it was incorrect to treat all fourteen schedules as equally expensive.
