# MoJev byte-level tokenizer allocations

Removing per-byte string construction and per-merge slice allocation cuts
MoJev request allocations by roughly 40–64% on the four measured workloads.
Responses remain exactly equal. The shared byte-level tokenizer microbenchmarks
run 39–45% faster; full-request latency has no established improvement.

## Profile and change

A fresh SIMD profile at `d99bd9d4` attributed 74.05% of CPU samples to
`gebpMicroKernel`. Tokenisation was a smaller CPU cost but accounted for
9,028 of 18,950 allocation objects cumulatively through `encodeByteLevel`
(47.64%). That profile used the four existing text workloads, one warm-up and
five measured calls each, after constructing the model. Allocation attribution
used `runtime.MemProfileRate=1`; those instrumented times are excluded from
performance comparisons. Profile totals include harness work and some sampled
constructor objects, so whole-request benchmark counts are reported separately.

`loader/tokenizer/tokenizer.go` now:

- Initialises 256 immutable byte-symbol strings under the existing `sync.Once`.
- Reuses a call-local symbol slice across pre-tokenised pieces.
- Merges adjacent symbols in place, preserving the leftmost minimum-rank rule.
- Appends IDs directly into the request-owned result instead of allocating a
  temporary result slice for each piece.

The byte mapping, regex splitting, direct whole-piece vocabulary lookup,
merge ranking, unknown-symbol handling, special tokens and SentencePiece path
are unchanged. Mutable scratch remains call-local. No input text, BPE result or
unbounded cache is retained between requests. The shared symbol table contains
256 strings and their small payloads, independent of request size.

The refreshed allocation profile attributes 4,214 of 14,137 objects to
`encodeByteLevel`, a 53.3% reduction in its cumulative count. Remaining sources
include regexp pre-tokenisation, string joins/merges, JSON request decoding,
packing and owned outputs. GEMM remains the major measured CPU cost.

## Measurements

Host: Linux amd64, Go 1.26.3, Intel i7-12700, `GOMAXPROCS=6`. GPU runs use RTX 3060
and driver 580.173.02. Checkpoint/model execution is serial. The approved
checkpoint and tokenizer hashes are verified by the existing probe before load.
Weights SHA-256:
`eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50`.
Tokenizer SHA-256:
`06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`.

Synthetic byte-vocabulary benchmarks include warm-up and request-owned output.
Ten samples per version, `-benchtime=100ms`, compared with benchstat:

| Workload | Before µs | After µs | Allocations before → after | Bytes before → after |
|---|---:|---:|---:|---:|
| Short sentence | 4.406 | 2.444 | 58 → 25 | 1.520 → 1.099 KiB |
| Repeated long input | 662.7 | 383.5 | 8,857 → 2,587 | 310.7 → 168.2 KiB |
| Merge-heavy tokens | 890.8 | 542.7 | 13,337 → 4,123 | 422.0 → 180.7 KiB |

Time reductions are 44.53%, 42.13% and 39.08%, respectively (all p<0.001, n=10).
These are synthetic tokenizer timings, excluding model execution.

Two sequential before/after process pairs per backend each measured five full
calls per workload. First pair median request allocation counts:

| Workload | GPU before → after | SIMD before → after |
|---|---:|---:|
| Short, two choices | 664 → 381 | 658 → 377 |
| Short, eight choices | 1,012 → 588 | 1,000 → 577 |
| Two questions, two choices | 1,034 → 618 | 1,007 → 585 |
| Longer, two choices | 1,612 → 585 | 1,593 → 569 |

The reversed-order repeat reproduced the reductions: GPU after counts
381/588/618/585, SIMD 368/579/585/571. Every response matches its before result
exactly, including probabilities and usage. GPU medians remain roughly
37/40/72/70 ms; SIMD timings vary between processes and some increased. No
whole-request speedup, peak-RSS reduction or model-quality improvement is
established by this change. The preserved GPU baseline predates packed-only
SIMD storage; its GPU/tokenizer implementation is the same as the pre-change
comparison. SIMD baseline uses the packed-only implementation.

## Tests and limits

A separate slow reference retains the old allocation-heavy BPE procedure.
Fixed and seeded random corpora compare both byte-level pre-tokenisation modes,
Unicode, malformed UTF-8, whitespace, repeated-pair ties, merge chains,
whole-piece lookup and unknown merged symbols. Twenty-four simultaneous callers
check deterministic output and independent result storage. An exact BPE
allocation assertion checks one joined lookup plus three merged strings with
preallocated scratch and output.

The initial full-tokenizer allocation ceiling was unstable under the race
detector because regexp's `sync.Pool` can discard entries there. That assertion
was replaced by the isolated BPE check; repeated full-tokenizer allocation
measurements remain in the benchmarks. The initial delegated test draft also
required fixing literal newlines in Go string literals before it compiled.

All 18 independently pinned released MoJev tokenizer cases and selected
packing/response/control tests pass under `-race -count=3`. Token IDs, fixtures
and numerical tolerances are unchanged. Tokenizer races pass ten repetitions;
the whole-tree NVIDIA-disabled race suite exits zero. Vet/build, docs/layout
checks and Linux ARM64/RISC-V builds and tokenizer test-binary cross-builds pass.
Foreign binaries were not executed. Focused read-only review found no semantic,
concurrency or ownership issue.

Default tokenizer statement coverage is 81.3%; changed functions are
`encodeByteLevel` 92.9%, `bpeMerge` 100% and `getByteEncoder` 100%. The unexecuted
empty-piece guard is defensive: current pre-tokenisation does not emit empty
pieces. Shared-loader scope extends beyond MoJev; the whole-tree model-free
suite covers other callers, but their released checkpoints were not executed
for this change.

```sh
go test -race ./loader/tokenizer -count=10
go test ./loader/tokenizer -run '^$' -bench '^BenchmarkByteLevel' \
  -benchmem -count=10 -benchtime=100ms

GO_PHERENCE_DISABLE_NVIDIA=1 \
  GO_PHERENCE_MOJEV_CHECKPOINT_DIR=/dev/shm/mojev-checkpoint \
  go test -race ./model/mojev \
  -run 'Tokenizer|TextPacking|TextDecision|TextControl' -count=3 -v

GO_PHERENCE_DISABLE_NVIDIA=1 go test -race -p=2 ./... -count=1 -timeout=180s
```

Evidence, profiles, raw timings and probe sources are under
`/workspace/tmp/mojev-refresh-20260926`. The public response benchmarks execute
real CPU/GPU inference; the synthetic BPE reference does not use model weights.
Native foreign-architecture inference, held-out quality/calibration and
hours-long service retention remain unqualified. `RuntimeReady=false`.
