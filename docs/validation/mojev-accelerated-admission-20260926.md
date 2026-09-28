# MoJev accelerated capacity and retained memory

Both accelerated scorers passed ordinary bounded stress runs at 512 tokens per
candidate path and 4096 total encoded tokens. This adds tests and evidence to
`4f9c8142`; runtime code and numerical behaviour are unchanged.

## Workloads and scope

Each process hash-checks and loads the existing approved checkpoint once, then
constructs one accelerator at capacity 512. The SIMD and NVIDIA runs are
separate. Host: Intel i7-12700, Linux amd64, Go 1.26.3, `GOMAXPROCS=6`, RTX 3060,
driver 580.173.02. Model revision
`0c8695b6252f4205907433d4e196a94f032e60c3`; SHA-256
`eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50`.

The opt-in `TestMoJevAcceleratedAdmission` exercises:

- The pinned base fixture, compared to its independent logits within `3e-4`.
- A 128-token state, 128-token question and two 256-token candidates: each
  isolated path is exactly 512 tokens.
- A 64-token state, 64-token question and 64 candidates of 62 tokens each:
  `64 + 64 + 64×62 = 4096` total tokens, executed in bounded candidate groups.
- Rejection of a 513-token path and a 4097-total-token request, including the
  expected error category and nil output.
- Three ordinary rounds alternating maximum path, maximum total and base
  requests, retaining all returned logits and independent copies.
- Eight concurrent callers sharing one scorer, alternating base and 512-token
  requests. Execution serialises over scratch. The 4096-token case is tested
  sequentially, not with eight simultaneous callers.
- Eight pre-canceled callers while the scratch lock is held; active cancellation
  at a deterministic execution poll and recovery at maximum path size.

The large inputs use deterministic vocabulary-valid IDs, not held-out natural
language. Their baseline logits come from the same implementation, so they
qualify determinism, ownership and reuse only. Independent large-shape accuracy
is untested. Across separate ordinary runs, maximum SIMD/PTX differences were
below `1.8e-6` for base, `9.8e-7` for the 512-path case and `1.48e-6` for the
4096-total case; cross-backend agreement is supplementary evidence.

## Memory results

Every snapshot runs GC, then records heap, stacks, goroutines and `/proc` RSS.
`runtime.KeepAlive` retains the scorer and CPU weights through collection;
outputs and copies remain live until after the final retention snapshot. Timing
from these stress tests is not an optimisation benchmark because forced GC and
logging are part of the probe. A 32 MiB post-GC growth allowance is an explicit
regression ceiling, not a claim that 32 MiB of growth occurred.

| Backend / run | Warm live Go heap bytes | Maximum retained-round growth | Peak RSS KiB | Result |
|---|---:|---:|---:|---|
| SIMD ordinary, 3 rounds | 5,103,747,424 | 24,160 B | 6,035,188 | pass |
| NVIDIA ordinary, 3 rounds | 3,019,857,552 | 3,248 B | 6,096,200 | pass |
| NVIDIA final ordinary, 3 rounds | 3,019,858,832 | 3,136 B | 6,044,948 | pass |
| SIMD race, 1 round | 5,103,764,816 | 19,616 B | 15,340,716 | pass |
| NVIDIA race, 3 rounds | 3,019,862,592 | 3,584 B | 17,930,076 | pass |

Ordinary recovery snapshots are below the warm live-heap baseline. Both backends
return to two observed goroutines; no projection workers remain parked. The
GPU scorer stays at exactly 2,057,538,816 device bytes until close, then reports
zero. Driver free-memory measurements increase after close, though global free
memory can also change due to unrelated device use. Host memory includes the
CPU copy of weights and SIMD packing; it is much larger than output retention.

The first ordinary SIMD run preceded adding the caller start barrier and the
maximum-shape cancellation check. A final one-round ordinary run with those
checks and exact rejection messages passed (peak RSS 6,055,808 KiB). Final NVIDIA
ordinary and bounded SIMD race runs include them. These are bounded tests, not
hours-long retention or arbitrary queued-caller admission.

## Race outcomes and timeout

The full three-round SIMD race run **timed out after 660 seconds** during the
third retention round. The stack was still computing; no data-race report
appeared before timeout. That run is not a pass. Its partial log is preserved
as `simd-race.log`.

A separately labelled one-round SIMD race run kept all sizes, eight callers,
cancellation and recovery checks and passed in 520 seconds. The NVIDIA
three-round race passed in 89 seconds. Race-instrumented peak RSS is a separate
resource budget, about 14.6 GiB SIMD and 17.1 GiB NVIDIA, compared with about
5.8 GiB ordinary peaks.

## Model-free boundaries and commands

Offline tests cover exact/over 4096 total tokens, 256/257 questions, 64/65
candidates, empty/negative tokens, early vocabulary/capacity rejection before
model work, and 512-token GPU metadata. The scoped limit run covers
`validateBranchLocalText` at 95.5%; it is not whole-package coverage.

```sh
# Default skips released stress work; limit tests remain offline.
GO_PHERENCE_DISABLE_NVIDIA=1 go test -race ./model/mojev -count=3

# Run backends separately. Keep the existing approved checkpoint.
GOMAXPROCS=6 GO_PHERENCE_MOJEV_ADMISSION_BACKEND=nvidia \
  GO_PHERENCE_MOJEV_CHECKPOINT_DIR=/dev/shm/mojev-checkpoint \
  go test ./model/mojev -run '^TestMoJevAcceleratedAdmission$' -v -count=1 -timeout=240s

GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 \
  GO_PHERENCE_MOJEV_ADMISSION_BACKEND=simd \
  GO_PHERENCE_MOJEV_ADMISSION_ROUNDS=1 \
  GO_PHERENCE_MOJEV_CHECKPOINT_DIR=/dev/shm/mojev-checkpoint \
  go test -race ./model/mojev -run '^TestMoJevAcceleratedAdmission$' -v -count=1 -timeout=600s
```

`GO_PHERENCE_MOJEV_ADMISSION_ROUNDS` defaults to 3 and accepts 1–3. JSON reports
include the actual round count; `GO_PHERENCE_MOJEV_ADMISSION_REPORT` selects an
output file. Logs and JSON live in `/workspace/tmp/mojev-admission-20260926`.

The default whole-tree CPU race suite exits 0; vet, build, docs/link checks
and ARM64/RISC-V test-binary cross-builds pass. Foreign test binaries were not
executed.

The focused review checked fixture arithmetic, live roots and retained outputs.
It identified limits now stated above: large cases are self-referenced and
concurrent execution excludes the maximum-total case. The double GPU close is
intentional because the scorer's public close contract is idempotent.

These tests qualify a bounded single-scorer execution envelope, with 256-question
limits validated only model-free. Independent full-context accuracy, concurrent
4096-token calls, prolonged retention, non-amd64 runtime execution, wider
failure coverage and held-out task quality remain open. `RuntimeReady=false`.
