## Needle allocation and pprof audit

The new Needle paths were spending a substantial part of their time building temporary execution objects rather than doing model arithmetic. Profiling the cached decoder showed `mallocgc` at 42.6% cumulative CPU; the packed matrix benchmark spent 88.5% of sampled CPU unpacking codebook indices. Both costs were measurable without changing model numerics.

This audit covers the new `model/needle`, `loader/needle`, `cmd/needle`, checked matrix/CQ runtime additions, and the ARM kernel corrections they exercise. It combines source inspection, CPU and allocation profiles, fixture benchmarks, a pinned released-model prompt, and native Intel/ARM correctness tests. It is not an exhaustive safety audit or a claim that every new line has an independent review. The delegated source/performance review timed out; no coverage is attributed to it.

## Changes supported by measurements

Inference no longer creates discarded backward closures. Contiguous slices are read-only views; simple layouts are copied directly instead of building integer index maps. The cached decoder groups float buffers and tensor descriptors into per-step arenas, with separate physical-capacity accounting. Arenas are not pooled across requests, and returned logits are copied so a caller does not retain a whole arena block.

Sinkhorn inference now runs the same twenty row/column log-normalisation passes in one matrix. It preserves the reference arithmetic order rather than constructing transpose/log-softmax nodes for each pass. Training retains the differentiable path. Vector addition/scaling use existing SIMD runtime operations, while training slice/broadcast gradients use SAXPY instead of copied gather-index tables.

CQ2/CQ4 unpacking uses a validated 256-entry byte lookup table per packed matrix. This replaces variable-width chunk assembly in the measured hot loop; dot accumulation still uses SIMD. The table costs 4 KiB per matrix and is included in `Bytes()`. Scalar unpacking and Walsh butterflies remain, so this is not an assertion of full vectorisation.

LoRA merge and full-model updates no longer copy every immutable base tensor twice. Derived models share private unchanged storage and own changed tensors; the public checkpoint API still returns copies. Safetensors loading uses one 64 KiB decode buffer instead of allocating a whole encoded tensor beside its FP32 output. Admission now checks **decoded FP32 bytes** as well as encoded bytes, fixing F16/BF16 expansion that could exceed the advertised tensor budget. A header-only regression rejects that case without allocating a large fixture.

A trial that allocated a throwaway tensor before obtaining an arena descriptor was discarded after benchmarks showed increased memory. The final path allocates only the descriptor it uses.

## Measurements

Intel Core i7-12700, Linux amd64, Go 1.26.3. Final fixture measurements below use two 500 ms runs with `-benchmem`; before numbers are the recorded baseline runs. Normal variation and differing profiler overhead mean allocation counts are the stronger comparison than a single timing.

| Workload | Before | Final | Allocation change |
|---|---:|---:|---|
| Tiny Needle 3, twelve cached tokens | 0.626 ms | 0.271--0.272 ms | 922,786 -> 458,246--458,250 B; 21,445 -> 1,824 allocations |
| Tiny full-prefix sequence | 2.408--2.426 ms | 1.231--1.280 ms | 4.39 MB -> 1.50 MB; 72,221 -> 14,471 allocations |
| CQ4 576x768 matrix/vector | 0.519 ms | 0.233 ms | unchanged 3,072 B / one allocation; +4 KiB retained lookup table |
| Tiny full-gradient pass | 0.310 ms | 0.243--0.250 ms | 501,765 -> about 382,800 B; 7,371 -> 6,084 allocations |
| Tiny LoRA-gradient pass | 0.365 ms | 0.293--0.299 ms | 534,558 -> about 415,600 B; 7,588 -> 6,301 allocations |
| Tiny archive, twelve cached tokens, decoded | 1.252 ms | 0.548--0.567 ms | 1.88 MB -> 0.890 MB; 43,814 -> 3,565 allocations |
| Tiny archive, same sequence, packed | 2.552 ms | 1.130 ms | 2.02 MB -> 1.11 MB; 43,526 -> 3,901 allocations |

Final loader benchmarks: the 58,650-byte fixture archive parses in 0.391--0.395 ms, about 121.6 KB and 654 allocations. A repeated Unicode/marker tokenizer encode/decode takes 0.0425--0.0433 ms, about 76 KB and 528 allocations. These numbers identify future work; no loader/tokenizer speedup is claimed.

### Released model, not a scaled-up fixture

The [released-model report][real] records the pinned Needle 3 archive, SHA-256, prompt and token IDs. Here `BenchmarkNeedleReleased` excludes model/decoder setup from its timing but ingests the same framed prompt on each iteration. `GOMAXPROCS=2`, five final iterations:

| Prompt pass | Earlier audit sample | Final | Final allocated bytes/allocations |
|---|---:|---:|---:|
| Decoded | 178.9 ms | 180.7 ms | 93,779,907 B / 36,312 |
| Packed hybrid | 337.9 ms | 316.3 ms | 97,788,867 B / 37,608 |

The earlier prompt passes allocated 120.2 MB and 124.2 MB respectively. Decoded timing did not improve reliably, so no warmed real-model speedup is claimed. Packed remains slower than decoded and stays opt-in.

Cold twelve-token CLI runs include loading: decoded 1.53 s / 1,016,448 KiB peak RSS; packed 2.07 s / 1,024,256 KiB. Earlier runs were 1.62 s / 1,093,116 KiB and 2.76 s / 1,093,632 KiB. All runs retained the same twelve generated IDs; this is a consistency check, not task-quality evaluation.

## Reading the profiles correctly

CPU timing and exhaustive allocation sampling were collected separately. The initial `-memprofilerate=1` experiment inflated the decoder timing to 41.8 ms/op and pushed allocation profiling itself above 90% cumulative CPU. That run is used only to locate allocations, never as a performance baseline.

The final cached CPU profile still puts about 18% cumulative time in Sinkhorn, principally scalar `Exp`/`Log`. The final released-model profile includes setup and both benchmark branches even though benchmark timing excludes setup: `CQMatrix.mulOne` is 20.6% flat / 32.8% cumulative, dense `SgemmNN` 19.4% flat, archive Walsh expansion 13.3% flat, and archive CQ decoding 21.4% cumulative. Its allocation profile includes archive buffers as well as repeated prompt passes. It must not be described as a pure warmed decoder profile.

Training allocation samples remain dominated by tape values, generic gathers and backward closures. Direct layout operations reduced the measured counts, but a persistent training arena/activation planner has not been implemented.

## Remaining hotspots and priorities

| Area | Finding | Follow-up, not claimed done |
|---|---|---|
| Packed CQ | Scalar lookup expansion and per-group SIMD dispatch remain the main cost; all decoded tensors are also retained | Fuse unpack/dot in architecture-specific kernels, evaluate packed-only storage, benchmark mixed 2-/4-bit real weights |
| Sinkhorn | Twenty exact log-normalisation passes spend time in scalar exp/log | Batch lanes/tokens or use validated vector math; do not silently replace reference numerics |
| Decoder | Per-step arena blocks are freed each step; parameter name construction remains visible | A bounded reusable session workspace and prebound parameter descriptors, preserving rollback and ownership |
| Training | Generic gather/index/backward nodes remain expensive | Typed operations and bounded activation/scratch planning; larger-model gradient profiles before changes |
| Cold archive load | Full CQ expansion/FP16 norms and simultaneous packed/decoded retention | Packed-only model layout or streaming tensor ownership; lower RSS without raising limits |
| Tokenizer | Heap candidates and merge nodes allocate per segment | Bounded reusable candidate storage; retain exact leftmost merge and marker rules |
| Heads | Hidden-state collection and repeated pooling allocate | Share one hidden-cell pass when multiple heads are requested; no such API added here |
| Model transforms | `SliceDepth` and public checkpoint export deliberately copy | Keep public ownership semantics; optimise only internal exclusive-ownership paths |

The remaining work is explicit rather than calling all code "SIMD-smoothed". SIMD is already used for dense products, dot products, vector add/scale/multiply and SAXPY accumulation; changing buffer/layout work has produced larger gains than merely adding more vector instructions.

## Reproduction and validation

```sh
GO_PHERENCE_DISABLE_NVIDIA=1 go test -race -p=2 ./... -count=1 -timeout=180s
go test ./model/needle -run '^$' -bench '^BenchmarkDecoderNeedle3/cached$' \
  -benchmem -benchtime=2s -cpuprofile=decoder.cpu -memprofile=decoder.mem
go test ./model/needle -run '^$' -bench '^BenchmarkNeedleTraining' \
  -benchmem -benchtime=2s -cpuprofile=training.cpu -memprofile=training.mem
go test ./backends/simd/runtime -run '^$' -bench '^BenchmarkCQMatrix' \
  -benchmem -benchtime=2s -cpuprofile=cq.cpu
go test ./loader/needle -run '^$' -bench '^BenchmarkNeedle' -benchmem
GO_PHERENCE_DISABLE_NVIDIA=1 GOMAXPROCS=2 \
  GO_PHERENCE_NEEDLE_PROFILE_MODEL="$PWD/checkpoints/needle3/needle3.cact" \
  go test ./model/needle -run '^$' -bench '^BenchmarkNeedleReleased$' -benchtime=5x -benchmem
go tool pprof -top decoder.cpu
go tool pprof -top -sample_index=alloc_space decoder.mem
```

Host race tests passed for 114 packages (54 without tests), along with vet/build and AVX2/FMA-disabled parity. Native CIX P1 runs passed 67 model, 47 loader, six CLI and 564 SIMD tests/subtests at low priority. ARM was not used for comparative timing. Upstream logits, head outputs, gradients, cache rollback and caller-owned data tests remain unchanged; no tolerance was widened. Full GPU qualification, frozen evaluation replay and unrelated service changes were not performed.

Profiles, benchmark text, raw pprof top tables and native logs are retained under `/workspace/tmp/needle-perf-audit/`; the paired call-flow SVG/HTML/JSON use the same CPU sample edge data. The report and downloadable profile bundle accompany this audit.

[real]: needle-packed-real-model-20260920.md
