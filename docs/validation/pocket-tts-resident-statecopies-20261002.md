# Pocket TTS resident host-state copy reduction

## Status

Removed redundant host copies from full-state validation and resident completed-boundary checks. Numerical arithmetic, device kernels, replay identities, coherent snapshot transfers and public snapshot ownership are unchanged. This is an allocation reduction, not a new device execution scope or native performance acceptance.

HEAD remains `a90aec30567d485784f2e04d1803c543c9f2817f`; training changes are uncommitted. The eight staged inference paths / 469 insertions remain unchanged. Both GPU backends were disabled. No model, GPU probe, recovery, listening or publication run occurred.

## Changes

- `FullTrainingState.Validate` now uses borrowed read-only validation maps instead of allocating zero parameter arrays and cloning M/V/EMA arrays only to discard them.
- Internal `validateTensorList` and `validateAdamList` check inventory, duplicates, shape and finite values without cloning. Existing `tensorListToMap` and `adamListToMap` still produce independently owned arrays after successful validation. Checkpoint load/conversion ownership remains unchanged.
- Both resident graph and optimiser-only backends compare live trainer values directly with their owned completed snapshot. They no longer call `State()` merely to validate a completed boundary.
- The exact boundary checks still cover topology/rebound live parameters, buffers, moments, enabled EMA, config and step. Equality is the explicit checkpoint/mutation contract, not a generated-signal acceptance test. Disabled EMA and optional step-zero moments retain the existing `State()` behavior.
- Initial resident state still starts from an owned validated `State()`. Device `Snapshot()`, full-state conversion, trainer `LoadState()` and checkpoint creation still transfer/copy coherent owned arrays. Uncertain native failures still retain owners and block retry/checkpoint/close.

No host arrays are inserted into the stored boundary by borrowing. The trainer remains single-owner and non-concurrent as before. No tolerance, hash gate or generated-output comparison was introduced.

## Evidence

Directory: `/workspace/exports/pocket-gpu-training-statecopies-20261002`.
Gate window: **2026-10-02 02:21:37–02:22:16 UTC**.

| Gate | Passed |
| --- | ---: |
| New cases repeated (`-count=10`) | 60 top-level + 170 subtests |
| Same cases with race detector | 6 top-level + 17 subtests |
| Affected host/mock regressions | 109 top-level + 129 subtests |

Final logs have zero failures/skips. Cases cover read-only validator admission and conversion ownership; changed parameter/buffer/M/V/EMA/config/step/topology; renamed or malformed moments/EMA; nonfinite values; optional initial moments; disabled EMA; reference equivalence to the previous owned-`State()` check; boundary rejection before another update. Existing resume, full-step/AdamW/EMA, state conversion, malformed/atomic restore and runner failure tests remain passing.

Coverage: tensor/moment validation and owned conversion helpers 100%; exact value helper 100%; live boundary helper 94.1%. Coverage is host/mock only, not native execution.

### Allocation evidence

- Tiny completed-boundary check: **275 allocations/op** for `State()` plus comparison versus **4 allocations/op** for live comparison. Ten-run allocation measurements passed repeatedly; the gate asserts reduction, not an exact platform-specific count.
- A 32,768-element parameter/M/V/EMA fixture: read-only `FullTrainingState.Validate` uses **1,152 B/op, 4 allocations/op**. A reconstruction of the previously discarded zero-array/moment/EMA-copy work uses **525,440–525,441 B/op, 8 allocations/op** over three 100-iteration measurements.

The comparison benchmark reconstructs discarded allocation work, not an entire prior trainer request. It omits some common validation work, so timing is not a request speedup claim. Device download counts, snapshot logical size, native memory and released-size request latency were not reduced or measured.

Affected-package vet, `go build ./...`, Linux ARM64/RISC-V pockettrain builds, layout checks and diff checks passed. No kernel changed, so prior offline kernel compilation gates were not repeated.

### Review limits

A self-contained independent delegate review timed out. No independent source approval is claimed. An initial compile error caused by narrowing the initial owned snapshot's local scope was corrected before passing gates; initial and final logs are retained. Offline reference tests and source inspection are not native numerical/resource/performance approval.

## Remaining work

Full resident snapshots and their owned conversion/load copies still happen after each update. Host graph/constants materialisation before pruning, dense embedding VJP, live allocations, inter-tape boundaries and diagnostic synchronization remain. Native strict-FP/FTZ/released-size resources, sustained throughput and new-voice/listening qualification require coordinated release of the existing holds.
