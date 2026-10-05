# Pocket TTS primary distillation backward–finish tape

## Implemented scope

`ptx-resident-primaryfused-hybrid-v1` and `vulkan-resident-primaryfused-hybrid-v1` combine primary mixed reverse-over-forward and target/noise/time-gradient assembly in one bound tape. Returned input derivatives remain raw. The objective scale applies once to resident parameter-gradient accumulation.

Training changes are uncommitted at HEAD `a90aec30567d485784f2e04d1803c543c9f2817f`. The staged inference index remains unchanged: eight paths and 469 insertions. Both GPUs were disabled during offline validation. No model/native GPU, recovery, listening or publication run occurred.

## Gradient contract

The primary tape receives raw mixed value/tangent seeds, endpoint input-only VJP and delta derivative. The graph retains selected-time index 1 and the existing device-time reverse-over-forward arithmetic. It then executes the previous finish utility operation order:

- Add primary and endpoint input VJPs.
- Multiply the sum by `s` for target and `1-s` for noise derivatives.
- Compute `ds = dot(sum, target-noise) + primaryDS - ddelta`.
- Compute `dt = primaryDT + ddelta` and pack both time derivatives.

Condition, target, noise and time derivatives are not multiplied by the parameter objective scale. Step reduction still scales raw distillation condition/target/logvar contributions once. All flow parameter roots accumulate with the caller's `parameterScale`, including zero-scale calls. Endpoint parameter/condition/time stop-gradient behavior is unchanged.

Full-source parameter inventory and byte admission precede pruning. Public builders own inputs/constants; private synchronous builders borrow only named parameter sources. No cross-tape cache or storage reuse was introduced. Older scopes keep their separate primary backward and finish calls. Public minimal-LSD result fields, losses and raw derivatives are unchanged.

The new scope inherits endpoint/diagonal/backbone fusion, device norm/AdamW/EMA, state checks, lifetime and positive-zero sharing. CLI factory coverage and exact `resident-primaryfused-backend` replay-hash tests run offline without native construction.

## Offline accounting

Tiny complete-step fixture, two valid flow rows:

| Measure | Endpoint fusion | Primary/finish fusion |
| --- | ---: | ---: |
| Tapes | 27 | 25 |
| Boundary tensor downloads | 72 | 62 |
| Commands | 3,716 | 3,694 |
| Conservative logical tape bytes | 139,536 | 139,296 |
| Summed planned owned tape bytes | 31,648 | 31,536 |

Each valid row replaces two tapes/nine boundary downloads with one tape/four downloads. Unused primary output/tangent, input/time intermediates stay in graph. All parameter accumulation counts and full snapshots are unchanged. Native latency, peak memory and complete-request resource use are unmeasured.

## Verification

Evidence: `/workspace/exports/pocket-gpu-training-primaryfused-20261002`.
Script: `/workspace/tmp/pocket-primary-gate.sh`.
Final gate window: **2026-10-02 06:05:09–06:05:54 UTC**.

| Gate | Passed |
| --- | ---: |
| New tests, ten repetitions | 180 top-level + 300 subtests |
| New tests, race detector | 18 top-level + 30 subtests |
| Affected host/mock regressions | 299 top-level + 462 subtests |

Final logs contain zero failures/skips. References include standalone device-time mixed flow plus finish branch tapes over s=0/.3/1, t=s/1, zero/nonzero seeds, every input/time/parameter VJP; independent CPU mixed-AD with scalar F64 finish checks; existing pinned PyTorch mixed-AD/minimal-LSD/complete-step regressions with unchanged tolerances.

PTX/Vulkan mock full-step objective endpoints, padding, clipped multistep P/M/V/EMA and checkpoint/resume pass. Direct tests capture accumulated gradients: each equals its raw graph gradient times parameter scale 0/.25/1, and returned input derivatives remain identical across those scales. All roots are touched once. Snapshot at an incomplete gradient boundary is rejected.

Admission covers full-source exact budgets, public ownership/private parameter-only borrowing, invalid topology/time/delta/seed/finish sources, scope/owner mutation and parameter scale. Dispatch/malformed/nonfinite output failure tests block retry/checkpoint/ordinary close and retain uncertain owners.

Affected-package vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain builds, formatting, model-layout and diff checks passed. No kernel changed; compiler gates were not repeated. Scoped capability/public/private builder/result guard coverage is 100%; evaluator 89.5%. Earlier resident validation intercepts some defensive count/alias branches; its 95% helper target stays open. Whole-package/native coverage is not established.

### Development corrections and review

An independent-reference test initially called a nonexistent mixed-AD method and was corrected to `ForwardTimeJVPBackward`. A scale-policy test initially tried to snapshot an incomplete gradient boundary; the snapshot correctly failed. The test now captures accumulated gradients at the mock driver and explicitly checks snapshot rejection. Development logs are retained. No numerical tolerance changed.

The mathematical delegate timed out after 10 seconds. Independent source approval has not been obtained. Native strict-FP/FTZ, released-size resources, sustained performance and voice quality have not been accepted; GPU and publication holds are unchanged.

## Remaining boundaries

Primary/endpoint forward prediction admission, endpoint-to-primary mixed-seed handoffs, initial diagonal prediction, transformer/flow handoffs, forward EOS scalar, metadata/nonzero constants, dense embedding VJP, diagnostic synchronisation and full snapshots/live host copies remain. Overall implementation and native/resource/voice acceptance remain active.
