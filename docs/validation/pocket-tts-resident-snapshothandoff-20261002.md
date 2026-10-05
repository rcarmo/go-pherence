# Pocket TTS resident snapshot ownership handoff

## Status

Removed the intermediate parameter/M/V/EMA clone between a completed resident snapshot and the backend's owned host boundary. Public checkpoint conversion still returns independent arrays. Device snapshot transfers, trainer copies, numerical arithmetic and replay identities are unchanged.

HEAD remains `a90aec30567d485784f2e04d1803c543c9f2817f`; training changes are uncommitted. The eight staged inference paths / 469 insertions remain unchanged. Both GPU backends were disabled; no model, native GPU, recovery, listening or publication run occurred.

## Ownership contract

- Private `takeFullStateFromResidentSnapshot` is used only by the resident graph and optimiser-only completed-update paths. Inputs must be fresh, uniquely owned host arrays returned by `ResidentState.Snapshot()`, never arbitrary/user-supplied snapshots or retained mutable aliases.
- It runs the same template/snapshot/config/step/inventory/name/shape validation as public conversion, copies immutable template buffers and constructs the full-state lists around snapshot arrays.
- After complete successful conversion, it clears the source snapshot and returns the new owner. On conversion failure, the source is not consumed or mutated. Consumption occurs before `LoadState`: a later load failure is terminal in both backends and cannot retry from the consumed source.
- `LoadState` still copies parameters and owns cloned moments/EMA separately from the backend boundary. Exact mutation checks therefore remain effective. The private boundary never aliases live trainer arrays.
- Public `FullStateFromResidentSnapshot` still clones parameter/M/V/EMA and immutable buffers. Its public independence contract is unchanged. Initial checkpoint-to-resident conversion is unchanged.

This is a host-allocation reduction. It does not reduce device-to-host bytes, full snapshot size, graph materialisation or native request latency.

## Validation

Evidence: `/workspace/exports/pocket-gpu-training-snapshothandoff-20261002`.
Gate window: **2026-10-02 02:27:32–02:28:14 UTC**.

| Gate | Passed |
| --- | ---: |
| New cases repeated (`-count=10`) | 30 top-level + 110 subtests |
| Same cases with race detector | 3 top-level + 11 subtests |
| Affected host/mock regressions | 112 top-level + 140 subtests |

Final logs have zero failures/skips. Tests cover EMA enabled/disabled, public/private value parity, transferred slice provenance, source consumption, independent frozen buffers, separate live trainer storage, retained mutation guards and nil/malformed template/config/step/inventory/name/shape/nonfinite/negative-V rejection without consuming the source or mutating the trainer. Existing resident full-step, clipped multistep, resume, runner failure, boundary and checkpoint ownership/atomic restore regressions remain passing.

Targeted coverage: public conversion and private handoff 100%; shared conversion body 96.6%. Coverage is host/mock only. Affected-package vet, `go build ./...`, Linux ARM64/RISC-V pockettrain builds, model-layout and diff checks passed. No kernel changed; previous compiler gates were not repeated.

### Allocation evidence

- Tiny snapshot conversion: **295 allocations/op** for public independent cloning versus **59 allocations/op** for private ownership transfer.
- A 32,768-element parameter/M/V/EMA fixture, three 100-iteration benchmarks: public conversion **526,808–526,809 B/op, 17 allocations/op**; private move **2,520 B/op, 13 allocations/op**.

The byte difference is the four 32,768-element F32 arrays not cloned again. These benchmarks isolate conversion, excluding device download and trainer loading. Timing varies and is not a complete-request or native speedup claim.

### Review limits

A self-contained `judge` delegate found no concrete issue in the described design and identified two preconditions now documented in source: trusted uniquely owned snapshots only, and consumption at conversion success rather than load success. The judge did not inspect source files or run tests; this is not independent source approval or native numerical/resource/quality acceptance.

## Remaining work

Full snapshots still download each completed update and trainer loading still copies into independently owned live state. Broader boundary fusion, host graphs/constants built before pruning, dense embedding VJP, live allocation reductions, inter-tape transfers and diagnostic synchronization remain. Native resources/performance/voice qualification requires coordinated release of the existing holds.
