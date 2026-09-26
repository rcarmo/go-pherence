# MoJev architecture-specific projection padding

## Change

Qwen branch projections previously rounded token rows up to a multiple of 12,
the common multiple of the amd64 six-row and ARM64/RISC-V four-row GEMM tiles.
They now round up to the current architecture's tile size, exposed by
`simd.SgemmNTRowBlock`. Scratch sizing uses the same helper.

For example, a 41-row amd64 projection now computes 42 rather than 48 rows. A
42-row projection can bypass padding copies entirely. The production GEMM
kernel, packing, worker partitioning, reduction order and arithmetic are
unchanged. Only dummy projection rows disappear; attention, positions and
recurrent state still see exactly the original token count.

Baseline: `5c4a8db6`. Retain this change: it removes demonstrable redundant work,
passes bitwise old-path comparisons, and improves selected projection and request
measurements. The shared-host measurements do not establish a universal model
speedup. GPU execution remains on hold, and `RuntimeReady=false`.

## Numerical and ownership checks

- A test-only old-12 reference runs the same packed projection at its original
  padded height. Every real output is compared bitwise against the new path.
- Tests cover 26 row counts from 1 through 512, including tile boundaries and
  the 256/512 admission limits, with native dispatch and forced scalar dispatch.
- Both paths clear output before accumulation. Tests reuse NaN-filled scratch,
  alternate input signs, protect scratch/output boundaries and check error
  propagation for aliased internal buffers. Existing projection tests retain
  input/weight immutability, column tails and worker ownership checks.
- Geometry is checked for every count 1–512: padded rows cover the input, are
  tile-aligned, add less than one tile, and never exceed old-12 capacity.
- Warm projections allocate zero bytes/objects on amd64. The existing allocating
  RISC-V microkernel is not fixed or reclassified by this change.

The changed padding helper and `project` function have 100% scoped Go statement
coverage. Unchanged `projectRows` has 88.6% under these focused tests; package
coverage is only 2.2%. This is not an assembly-coverage or repository-wide claim.

A bounded delegated design review and final review found no concrete correctness
defect. They reviewed the supplied design/code description, not an independent
checkout. Residual gaps are native four-row-platform execution and full
constructor execution at every possible capacity. The geometry is exhaustive,
but released constructor runs here cover 256 and 512 only. Instantiating 512
copies of the released model was deliberately not used as a geometry test.

## Host and workload

Go 1.26.3, Linux amd64, reported Intel i7-12700, six available CPUs,
`GOMAXPROCS=6`, NVIDIA explicitly disabled. Baseline and candidate binaries are
preserved. The approved checkpoint revision remains
`0c8695b6252f4205907433d4e196a94f032e60c3`; each released process verifies all four
config, weight and tokenizer hashes before use. One checkpoint process ran at a
time. No GPU, frozen evaluation or model service was started.

Projection benchmarks use a 1024-input/3584-output packed projection, six scoped
workers and rows 6, 12, 18, 41, 42, 48, 60, 126, 128, 256 and 512. Packing, worker
creation, buffers and warmup are outside timing. Both binaries use old-12-sized
benchmark storage, so the measurements do not hide allocation differences.

Six interleaved ABBA samples per version, 200 ms per case:

| Real rows | Old-12 median | Architecture median | Comparison |
|---|---:|---:|---|
| 6 | 203.3 µs | 109.9 µs | −45.97%, p=0.002 |
| 18 | 341.6 µs | 266.4 µs | −22.03%, p=0.002 |
| 41 | 641.2 µs | 557.4 µs | Not significant, p=0.180 |
| 42 | 705.3 µs | 502.6 µs | −28.73%, p=0.002 |
| 126 | 1.590 ms | 1.466 ms | Not significant, p=0.485 |

All cases report zero B/op and allocs/op. Some unchanged-shape controls also
move significantly (rows 12 and 48), so the magnitude includes host variation.
The full 11-case geomean is −16.12%, not a model-level claim. Interleaved raw
samples are retained. The initial non-interleaved exploratory summary is also
retained, but its raw filenames were subsequently reused by the request
summarizer; it is not the acceptance dataset.

## Full requests, memory and profiles

Two independent ABBA matrices each contain six processes per version, four
existing capacity-256 workloads, one warmup and five timed calls per workload.
Loading is outside request timing. One-second load/pressure observations are
preserved. All 576 actual warmup/trial responses match exactly across baseline
and candidate; request hashes and names are also checked.

| Metric | First matrix | Repeat |
|---|---:|---:|
| Four-workload latency geomean | −5.65% | −4.54% |
| Short two-choice request | −12.00%, p=0.004 | Not significant |
| Short eight-choice request | Not significant | −6.36%, p=0.041 |
| Two-question request | −6.43%, p=0.009 | Not significant |
| Longer request | Not significant | Not significant |
| Allocated-byte geomean | +0.08% | +0.53% |
| Allocation-count geomean | −0.04% | −0.24% |

Individual significance is not stable between matrices; the repeat includes a
large timing outlier. Directional agreement supports retaining reduced work,
not a fixed service-level percentage. Both matrices remain in evidence.

One workload has a roughly 2% byte increase in each matrix, but it is a different
workload each time: two-question +2.03% initially, eight-choice +2.11% on repeat.
The same-workload increase does not reproduce. No whole-request allocation
improvement or equivalence is claimed. Warm projection tests isolate the changed
code at zero allocations; request-level variation remains an open observation.

Ordinary peak RSS ranges overlap: first baseline 4,986,368–4,992,000 KiB versus
candidate 4,987,520–4,992,000 KiB; repeat baseline 4,985,728–4,991,872 KiB versus
candidate 4,987,648–4,990,080 KiB. There is no measured RSS gain.

On amd64, capacity 256 and 512 allocate the same rounded scratch as before
(264 and 516 rows). Some other capacities avoid six scratch rows, logically
saving `6*(3584+6144)*4 = 233472` bytes. Four-row targets can save four or eight
rows. Those capacity/platform-specific sizes are geometry, not measured retained
heap or loading-peak results.

Separate normal-rate profiles, excluded from timing comparisons, show GEMM
samples falling from 19.62 s to 18.12 s; total sampled CPU falls from 25.85 s to
24.73 s. GEMM remains the main cost (75.90% before, 73.27% after). These single
profiles attribute work; they are not repeated benchmark evidence. Heap profiles
include `alloc_space`, `alloc_objects`, `inuse_space` and `inuse_objects`; loading
and packed weights still dominate. Loading code is unchanged and was not given
a new cold-cache qualification. Decode, training and multimodal workloads are
outside this text-only projection slice.

## Gates

- Focused old/new, geometry, projection and packed-only tests: count 10 passed.
- Focused projection races: count 10 passed; final whole-tree race includes the
  subsequently added projection-error test.
- Capacity-512 `TestReleasedGroupedTextScorer` race: passed in 353.05 s, Go test
  total 354.328 s, exit 0. Independent 4096-total-token reference, hidden rows,
  changed/reordered concurrent requests and retained-output ownership passed.
- Errors unchanged: logits `1.4007091522216797e-6`, changed logits
  `4.647299647331238e-7`, hidden `5.7220458984375e-5`; gates remain `3e-4` / `2e-3`.
- Post-concurrency live heap is 59,368 bytes below warm baseline. Race peak RSS
  11,120,352 KiB, zero swaps; not ordinary memory admission evidence.
- Whole-tree CPU race, `-p=2 -count=1 -timeout=180s`: exit 0, wall time 5m27.27s.
- `go vet ./...`, `go build ./...`, docs/layout checks: passed.
- Runtime/Qwen test cross-builds for Linux ARM64 and RISC-V: passed. Native
  execution on those architectures remains open; no cross-build is a pass there.

Hours-long retention, native foreign execution, held-out quality/calibration and
GPU qualification are not closed by these gates.

## Evidence and reproduction

Evidence: `/workspace/tmp/mojev-row-padding-20260926`, including preserved
binaries, original production file/overlay, interleaved projection samples, both
request matrices, actual-response checks, host observations, profiles and gate
logs. Binaries remain local with SHA-256 hashes; the downloadable bundle includes
reports, source/patches, profiles and raw observations.

```sh
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 go test ./model/qwen \
  -run '^$' -bench '^BenchmarkSIMDBranchProjectionPadding$' \
  -benchtime=200ms -benchmem -count=6
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 go test -race ./model/qwen \
  -run 'TestSIMDBranch(Padding|Projection|PackedOnly)' -count=10
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 \
  GO_PHERENCE_MOJEV_CHECKPOINT_DIR=/dev/shm/mojev-checkpoint \
  GO_PHERENCE_MOJEV_GROUPED_BACKEND=simd \
  GO_PHERENCE_MOJEV_GROUPED_CAPACITY=512 \
  GO_PHERENCE_MOJEV_GROUPED_REPORT=OUTPUT.json \
  go test -race ./model/mojev -run '^TestReleasedGroupedTextScorer$' \
  -v -count=1 -timeout=900s
```
