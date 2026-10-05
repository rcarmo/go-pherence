# Pocket TTS diagonal interpolation–forward tape

## Implemented scope

`ptx-resident-diagforward-hybrid-v1` and `vulkan-resident-diagforward-hybrid-v1` compose diagonal interpolation and initial flow forward in one resident-bound tape. The interpolated input feeds flow through a tensor ID. Three owned outputs preserve interpolated input, desired velocity and prediction for the existing loss-overflow admission and backward path. Forward parameter accumulation is disabled.

Training changes are uncommitted at HEAD `a90aec30567d485784f2e04d1803c543c9f2817f`. The staged inference index remains at eight paths and 469 insertions. Both GPUs were disabled during validation. No native/model, recovery, listening or publication run occurred.

## Arithmetic and ownership

The tape retains explicit F32 interpolation order: scale target by time, scale noise by one minus time, then add. Desired velocity is target minus noise. Flow uses `[time,time]` and selected tangent index -1. Its unused tangent no longer downloads. The prediction still crosses the host boundary for conservative loss admission; loss/backward scaling is unchanged.

`appendAdmittedFlowGraph` is extracted from the completed distillation-forward builder and shared by both composed forward paths. It remaps the input edge and optional read-only parameter identities without changing operation order. Complete interpolation and flow sources are admitted before remapping; the combined source/binding/byte ceiling is checked before pruning. Dead placeholders and parameter-gradient graphs remain charged before pruning. All parameter bindings are admitted, but non-accumulating execution roots only the three boundary values.

Public builders own constants. Private synchronous builders borrow only named parameter sources. Output guards reject nonfinite, malformed, aliased or caller-borrowed arrays. Failed dispatch or post-execution rejection blocks retry and checkpoint boundaries and retains uncertain owners. No cross-tape cache/storage reuse was introduced. Older scopes retain separate interpolation and flow calls; CPU defaults and distillation forward/backward are unchanged.

## Offline accounting

Tiny complete-step fixture, two valid flow rows:

| Measure | Distillation forward fusion | Diagonal forward fusion |
| --- | ---: | ---: |
| Tapes | 17 | 15 |
| Boundary tensor downloads | 50 | 48 |
| Commands | 3,688 | 3,684 |
| Conservative logical tape bytes | 132,952 | 132,904 |
| Summed planned owned tape bytes | 28,536 | 28,472 |

Each valid diagonal row replaces two tapes/four downloads with one tape/three downloads. Parameter accumulation counts, diagnostics and full snapshots are unchanged. These are mock/logical measurements; native latency, peak memory and complete-request performance are unmeasured.

## Verification

Evidence: `/workspace/exports/pocket-gpu-training-diagforward-20261002`.
Script: `/workspace/tmp/pocket-diagforward-gate.sh`.
Gate window: **2026-10-02 06:25:43–06:26:35 UTC**.

| Gate | Passed |
| --- | ---: |
| New tests, ten repetitions | 170 top-level + 260 subtests |
| New tests, race detector | 17 top-level + 26 subtests |
| Affected host/mock regressions | 340 top-level + 542 subtests |

Final logs contain zero failures/skips. All three outputs match standalone interpolation/device-time flow and independent ordinary CPU flow/scalar interpolation at times 0/.3/1. Existing independent F64 loss, pinned PyTorch mixed-AD/minimal-LSD/complete-step, objective endpoints, padding, clipped multistep P/M/V/EMA and checkpoint/resume regressions pass without tolerance changes.

PTX/Vulkan mock tests verify no forward accumulation and three boundary roots. Tests cover exact/full-source budgets, invalid topology/time/input values/shapes, public ownership/private parameter-only borrowing, dead-invalid source admission, result finite/shape/alias guards, scope/owner mutation and dispatch/post-execution failure retention. Standalone and composed distillation-forward regressions cover the shared helper.

Affected-package vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain builds, formatting, model-layout and diff checks passed. No kernel changed; compiler gates were not repeated. Scoped capability/public wrapper/result guard coverage is 100%, builder 91.7%, evaluator 88.2%, shared composition helper 90.9%. Defensive remap/inventory guards and malformed driver-return branches were not all reached; 95% helper targets stay open. Whole-package/native coverage is not established.

### Development correction and review

Initial accounting expected unchanged downloads. The unused initial-flow tangent accounts for one fewer download per valid row; expectations now match the measured boundary reduction. The initial log is retained and no numerical tolerance changed.

The mathematical delegate timed out after 10 seconds. Independent source approval has not been obtained. Native strict-FP/FTZ, released-size resources, sustained throughput and voice quality have not been accepted. GPU and publication holds are unchanged.

## Remaining boundaries

Prediction/admission values, transformer/flow handoffs, forward EOS scalar, graph metadata/nonzero constants, dense embedding VJP, diagnostic synchronisation and full snapshots/live host copies remain. Overall implementation and native/resource/voice acceptance remain active.
