# MoJev GPU host-weight lifetime

The NVIDIA scorer releases about 1.99 GB of live host heap when the caller drops
its original CPU scorer. GPU inference keeps the shared embeddings and scorer
head, but no longer retains the uploaded CPU encoder. The caller's CPU scorer
remains usable. Successful `Close()` also releases the host embedding/head
references.

## Change and ownership

`NewNVIDIATextScorer` uploads from the original validated CPU scorer and retains
a separate host view containing only `head`, `embedding`, `meta` and `eps`.
It neither copies these payloads nor mutates the caller's scorer. Keeping the
original CPU scorer alive necessarily keeps its encoder weights alive too.

Text tokenisation and answer assembly now use a weight-independent helper.
`ScoreTextContext` checks readiness under the scorer's context-aware mutex,
then releases it for preprocessing. `ScoreEncodedContext` checks readiness
again under that mutex before inference. A concurrent close can therefore
reject an in-flight text request without returning partial answers or reading
a released host view. Scratch and device execution remain serialised.

`Close()` clears the host view after successful module cleanup. A failed
module close still disables inference and preserves ownership for a retry.
This failure path was reviewed; scorer-level driver-failure injection was not
added. Existing runtime module tests exercise retry after failed synchronisation
and unload. No CUDA kernels, numerical operations or tolerances changed.

## Measurement

Baseline: `0d13e058`, Linux amd64, Go 1.26.3, Intel i7-12700, RTX 3060,
NVIDIA driver 580.173.02, `GOMAXPROCS=6`. One checkpoint/backend process ran at a
time. Approved checkpoint revision: `0c8695b6252f4205907433d4e196a94f032e60c3`.
Weights SHA-256:
`eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50`.
The probe checks configuration, weights and both tokenizer hashes before loading.

The probe constructs both scorers, measures after GC while explicitly keeping
the CPU scorer live, drops it, calls `debug.FreeOSMemory`, measures again, then
runs the same four text workloads. Each workload has one warm-up and five
measured calls. Forced GC/scavenging occurs outside timed requests; production
code adds no GC calls. Two final processes reproduced the memory reduction.

| Measurement | Baseline | Final | Final repeat |
|---|---:|---:|---:|
| Live heap, both scorers (bytes) | 3,085,706,704 | 3,085,707,104 | 3,085,679,792 |
| Live heap, GPU only (bytes) | 3,085,705,576 | 1,092,104,208 | 1,092,076,320 |
| Live heap, after requests (bytes) | 3,017,814,664 | 1,024,174,816 | 1,024,258,320 |
| Whole-process peak RSS (KiB) | 6,157,560 | 6,144,828 | 6,165,196 |

Sampled request-time RSS fell from 3,124,628–3,126,024 KiB to
1,177,672–1,179,156 KiB (about 2.98 GiB to 1.12 GiB). Peak loading RSS is
unchanged within run variation: the original weights and upload temporaries
still coexist during construction. GPU residency and scratch capacities are
unchanged. The new host view costs one small constructor allocation.

All four example responses match the baseline exactly; the probe also checks
repeatability within each workload. Median request times in milliseconds:

| Workload | Baseline | Final | Final repeat |
|---|---:|---:|---:|
| Short, two choices | 38.409 | 37.242 | 36.265 |
| Short, eight choices | 41.496 | 39.752 | 40.237 |
| Two questions, two choices | 75.554 | 72.060 | 71.997 |
| Longer, two choices | 72.646 | 69.316 | 69.650 |

These sequential memory probes do not establish a latency improvement. No
benchstat significance test was used for this retention-only change.

## Tests and review

`TestNVIDIATextScorerHostLifetime` verifies that:

- Upload leaves the original CPU scorer usable against the pinned base logits.
- A weak pointer to its encoder root clears after the constructor helper returns
  and GC runs, while the GPU scorer stays live.
- GPU logits still pass after encoder collection.
- Successful close permits collection of the host view, head and embedding
  backing allocation, while the closed GPU scorer stays live; residency is zero
  and repeated close succeeds.

CPU/GPU base-logit maximum errors were `1.28150e-6` / `2.20537e-6`, below the
unchanged `3e-4` gate. The released GPU regression and lifetime test passed under
the race detector in one serial process (155.2 seconds including load and
instrumentation). The model-free close test races 32 text callers with close;
it exercises rejection with an empty host scorer, not successful GPU inference.
The separate released regression exercises real scoring and concurrency.

Model-free orchestration tests cover successful answers, callback errors,
malformed logits, request validation, packing errors, cancellation boundaries,
and readiness-lock cancellation. Repeated focused races pass. The extracted
helper and CPU wrapper each reach 100% statement coverage in the focused test.
Default package coverage includes unexecuted opt-in hardware/weight paths and
is not a complete hardware coverage measure.

A first focused review identified host references surviving successful close;
that finding was fixed and tested. A second read-only review found no issues.
It did not execute hardware tests or inject scorer-level driver failures.
Weak-root checks do not enumerate every individual encoder allocation; the
heap measurements independently quantify the retained-memory change.

Completed checks include the whole-tree NVIDIA-disabled race run (exit 0),
affected package races repeated three times, vet/build, docs/layout checks and
Linux ARM64/RISC-V cross-builds. The first whole-tree shell invocation timed out;
the longer rerun completed. Cross-built binaries were not executed.

```sh
GOMAXPROCS=6 GO_PHERENCE_MOJEV_NVIDIA=1 \
  GO_PHERENCE_MOJEV_CHECKPOINT_DIR=/dev/shm/mojev-checkpoint \
  go test -race ./model/mojev \
  -run '^(TestNVIDIATextScorerHostLifetime|TestMoJevAcceleratedReleased|TestNVIDIATextCloseRegression|TestTextOrchestration)$' \
  -v -count=1 -timeout=300s

GO_PHERENCE_DISABLE_NVIDIA=1 go test -race -p=2 ./... -count=1 -timeout=180s
GO_PHERENCE_DISABLE_NVIDIA=1 go test -race \
  ./model/mojev ./model/qwen ./backends/nvidia/runtime -count=3
go vet ./...
go build ./...
make docs-check
GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build ./...
GOOS=linux GOARCH=riscv64 CGO_ENABLED=0 go build ./...
```

Raw probes, logs and response comparisons are under
`/workspace/tmp/mojev-gpu-host-20260926`. Three preceding kernel experiments
(paired MLP projections, warp-local attention and precomputed recurrence
parameters) did not establish consistent end-to-end gains and were reverted.
Their experimental sources and timings remain outside the repository under
`/workspace/tmp/mojev-gateup-20260926`.

Prolonged retention, full three-round SIMD admission race, native foreign
execution and held-out quality/calibration still need qualification.
`RuntimeReady=false`.
