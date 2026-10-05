# Pocket TTS distillation endpoint loss–VJP tape

## Implemented scope

`ptx-resident-endpointfused-hybrid-v1` and `vulkan-resident-endpointfused-hybrid-v1` combine minimal-LSD loss, negated endpoint seed, endpoint input-only VJP and mixed primary seeds in one resident-bound tape. The endpoint forward output still supplies the existing overflow admission. No endpoint parameter, condition or time gradient accumulates.

Training changes are uncommitted at HEAD `a90aec30567d485784f2e04d1803c543c9f2817f`. The eight-path staged inference index is unchanged at 469 insertions. Both GPUs were disabled during validation. No model/native GPU, recovery, listening or publication run occurred.

## Stop-gradient and scaling

The graph computes residual `desired - endpoint`, where desired is primary velocity plus delta times its selected-time tangent. LSD loss produces raw statistics and residual seed `dr`. The endpoint is reevaluated with `-dr` and zero tangent seed. Only its input VJP `dx` enters the primary seeds:

- `gv = dr + delta * dx`;
- `gt = delta * dr`;
- `ddelta = dot(dr, primaryTangent) + dot(dx, primaryVelocity)`.

The graph preserves the branch tape's dot/reduction and multiply/add order. Resident execution explicitly uses `accumulate=false`; all source parameter bindings still undergo full admission, while pruning drops unused endpoint parameter/condition/time outputs. The primary mixed flow backward retains its existing objective-scaled parameter accumulation. Raw condition/target/logvar derivatives and statistics retain later step-reduction scaling.

Public builders own constants. Private builders borrow only named parameter sources synchronously. Endpoint prediction is admission-only; the immutable endpoint is recomputed from condition/time/input. Source/inventory/byte ceilings, lifetime/positive-zero planning and terminal failure retention are unchanged. Legacy scopes retain separate loss, negation, endpoint VJP and mixed utility tapes. Distillation public velocity/tangent/endpoint/residual fields are unchanged.

## Offline accounting

Tiny complete-step fixture, two valid flow rows:

| Measure | Diagonal fusion | Endpoint fusion |
| --- | ---: | ---: |
| Tapes | 33 | 27 |
| Boundary tensor downloads | 86 | 72 |
| Commands | 3,956 | 3,716 |
| Conservative logical tape bytes | 144,880 | 139,536 |
| Summed planned owned tape bytes | 33,064 | 31,648 |

Each row replaces four tapes and twelve downloads with one tape and five downloads. Endpoint input VJP, mixed primary seeds, raw stats and delta derivative are returned; prediction seeds and unused flow boundaries stay in graph. Accumulation counts and snapshots are unchanged. Native speed, peak memory and request performance are unmeasured.

## Verification

Evidence: `/workspace/exports/pocket-gpu-training-endpointfused-20261002`.
Script: `/workspace/tmp/pocket-endpoint-gate.sh`.
Gate window: **2026-10-02 05:57:36–05:58:23 UTC**.

| Gate | Passed |
| --- | ---: |
| New tests, ten repetitions | 180 top-level + 310 subtests |
| New tests, race detector | 18 top-level + 31 subtests |
| Affected host/mock regressions | 285 top-level + 432 subtests |

Final logs contain zero failures/skips. Tests compare standalone loss/negation/device-time endpoint backward/mixed tapes for times 0/.4/1, delta 0/time and logvar -2/0/2. Independent ordinary CPU flow VJP plus scalar F64 loss/mixed references pass existing tolerances. Tests cover public ownership, exact/full-source budget, invalid finite/time/delta/logvar/shape/overflow inputs and malformed/aliased result guards.

PTX/Vulkan mock tests demonstrate zero endpoint accumulation with nonzero input VJP, zero/nonzero delta, unchanged all-gradient full-step/update/EMA, objective endpoints, padding, clipped multistep state and checkpoint/resume. Dispatch and post-execution failures block retry/checkpoint and retain owners. Pinned PyTorch minimal-LSD/mixedAD and complete-step regressions remain passing.

Affected-package vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain builds, formatting, model-layout and diff checks passed. No kernel changed; compiler gates were not repeated. Scoped capability/public/private builder/result guard coverage is 100%, resident evaluator 88.2%. Earlier resident output validation intercepts some defensive evaluator branches; its 95% helper target remains open. Whole-package/native coverage is not established.

The mathematical delegate timed out after 10 seconds. Independent source approval has not been obtained. GPU strict-FP/FTZ, released-size resource limits, sustained performance and voice quality have not been accepted. Holds are unchanged.

## Remaining boundaries

Primary distillation backward/finish, primary and endpoint forward prediction admission, transformer/flow handoffs, forward EOS scalar, metadata/nonzero constants, dense embedding VJP, diagnostic synchronisation and full snapshots/live host copies remain. Overall implementation and native/resource/voice acceptance remain active.
