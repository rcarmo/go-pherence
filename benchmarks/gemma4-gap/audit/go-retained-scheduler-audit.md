# Retained Go prompt scheduler audit

Workload: real Gemma 4 E4B Q4_0 GGUF, 124-token prefill, `taskset -c 0-5`, `GOMAXPROCS=6`.

Trace run:

```bash
taskset -c 0-5 env GOMAXPROCS=6 \
  GO_PHERENCE_GEMMA4_GAP_REAL=1 \
  GO_PHERENCE_GEMMA4_PREFILL_ONLY=1 \
  go test -trace=benchmarks/gemma4-gap/audit/go_retained_124_prefill_scheduler.trace \
  ./model -run '^TestGemma4RealCPUGap124x48$' -count=1 -v -timeout=10m
```

The trace run is attribution evidence, not an accepted throughput median. It measured 2.919579 s and 42.4719 prompt tok/s under shared-host conditions.

## Corrected utilisation

The dedicated prefill phase spans 2.919579 s. Summed `Running` time across the six Go Ps is 10.804739 CPU-s, so runtime utilisation is:

```text
10.804739 / (2.919579 * 6) = 61.68% of six Ps
```

The earlier 10.1% estimate divided the wrong quantities and is invalid. Unused capacity is 6.712733 CPU-s, or 1.118789 s at six-way saturation.

## Projection topology and worker count

`go_retained_projection_shapes.log` records the real model:

- 42 layers, hidden size 2560;
- every layer has Q, O, gate, up and down Q4_0 projections: 210 calls;
- layers 0--23 additionally have K and V Q4_0 projections: 48 calls;
- 258 Q4_0 batched projections total; and
- one final `262144x2560/Q6_K` LM-head projection.

The trace contains exactly 1,548 activation-quantisation worker goroutines (258 calls by six workers) and 1,554 GEMV worker goroutines (259 calls by six workers). `gemvRowsParallel` creates fresh goroutines with six static contiguous row shards for every call. The caller blocks in `WaitGroup.Wait`; it performs no shard. `quantizeQ8_0BatchTo` has the same fresh-worker/static-shard structure.

## CPU and completion balance

Derived artefacts:

- `go_retained_124_prefill_workers.tsv` -- worker creation time, traced running time and final running timestamp;
- `go_retained_124_prefill_worker_balance.log` -- per-call span, utilisation and completion skew; and
- `go_retained_124_prefill_shape_utilisation.log` -- aggregate attribution by matrix shape.

Worker running time sums to 9.457 s in projection and 0.649 s in quantisation. Approximate parent wait spans are 1.996 s and 0.170 s respectively. The static projection shards therefore use about 79.0% of six-way capacity over their wait spans; quantisation uses about 63.6%.

Most individual GEMV calls are well balanced: median per-call utilisation is 94.2% and median worker skew is 9.0%. The aggregate loss is in long-tail completions. In particular:

| Shape | Calls | Wait span | Worker running | Aggregate six-P utilisation |
|---|---:|---:|---:|---:|
| `10240x2560` | 84 | 1.005491 s | 5.038489 CPU-s | 83.5% |
| `2560x10240` | 42 | 0.635001 s | 2.828216 CPU-s | 74.2% |
| `2048x2560` | 35 | 0.107745 s | 0.476703 CPU-s | 73.7% |
| `2560x2048` | 35 | 0.109658 s | 0.449784 CPU-s | 68.4% |
| `512x2560` | 40 | 0.037577 s | 0.148821 CPU-s | 66.0% |

The idealised `sum(worker running)/6` lower bound is 1.576 s for GEMV and 0.108 s for quantisation. Relative to observed wait spans, static-shard tail removal has at most about 0.420 s and 0.062 s available in this trace. That is a useful optimisation ceiling, not a promised speed-up: traced `Running` intervals can include host pre-emption while assembly executes, and dynamic scheduling cannot recover all host contention.

`WaitGroup.Wait` time is not itself overhead: it mostly covers useful worker computation. Scheduler-delay profiles attribute only milliseconds to goroutine readiness, so persistent workers cannot close the prompt gap by startup removal alone.

## Frozen llama.cpp comparison

Frozen b607 differs in two relevant ways:

- a persistent graph thread pool means all existing threads, including thread zero, participate; and
- output rows are split into about four chunks per thread. Each thread starts with its own chunk, then claims remaining chunks through `ggml_threadpool_chunk_add`.

It also quantises activation planes cyclically across those existing threads. Go instead launches six workers per phase call, leaves the caller waiting, and fixes one contiguous row range per worker.

## Ranked scheduler hypotheses

1. **Caller participation plus overdecomposed dynamic row chunks.** This directly attacks both avoidable hand-off and observed static-shard tail time, and mirrors b607's four-chunks-per-thread mechanism. It preserves independent-row arithmetic and output order.
2. **Apply the same helper to 124-row activation quantisation.** The ceiling is smaller, but it removes the second static scheduler and preserves byte-for-byte independent output blocks.
3. **Persistent workers alone.** Previously regressed and the trace gives only millisecond-scale startup delay. Reject as a standalone gap-closing mechanism.

Even the idealised 0.482 s scheduler ceiling would move this slow trace only from 42.47 to about 50.9 prompt tok/s. Scheduling is therefore a coherent first batch and a useful prerequisite, but cannot reach the 89.4050 tok/s gate without a substantially faster fused projection kernel.
