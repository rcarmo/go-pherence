# Pocket TTS diagonal loss–flow backward tape

## Implemented scope

`ptx-resident-diagfused-hybrid-v1` and `vulkan-resident-diagfused-hybrid-v1` combine diagonal LSD loss seeds, objective scaling, flow backward and direct target-gradient assembly in one bound graph. Forward prediction remains a host boundary for conservative loss overflow admission. Distillation retains its existing execution and stop-gradient policy.

Training changes are uncommitted at HEAD `a90aec30567d485784f2e04d1803c543c9f2817f`. The eight-path staged inference index remains at 469 insertions. Both GPUs were disabled throughout validation; no native/model, recovery, listening or publication run occurred.

## Arithmetic and ownership

The loss graph writes `[rawSquare, loss, dLogVariance]` and the raw prediction seed. It scales that seed by `PEqual / validRows` once, then passes it to the existing ordinary flow reverse graph with a zero tangent seed. Resident accumulation uses unit scale. The direct target derivative keeps the existing two-operation order: scale flow input VJP by time, then subtract the already-scaled prediction seed.

Condition and target VJPs therefore include the diagonal objective scale. Loss statistics and log-variance derivative remain raw; step reduction applies the log-variance objective scale once. Zero objective still touches all flow parameter bindings. Scalar metrics, weighting, EOS/backbone, endpoint input-only VJP and primary distillation accumulation are unchanged.

`reverseIDs` is shared with the standalone flow builder, which still constructs caller-owned seeds in the same order. The loss source admission helper retains its finite/logvar and raw/exp-weighted reduction headroom. Prediction passed to the fused builder is used only for admission; immutable model, condition, time and input are recomputed inside the VJP graph. Public constants are copied and private named parameter sources remain read-only borrowed. Complete source/binding/byte admission precedes pruning.

The step selects the new capability only under the new scope and requires the same flow/loss/branch/step-reduction owner. Legacy scopes retain their standalone loss, scale, flow backward and target tapes. Replay includes `resident-diagfused-backend`, exact scope and runtime pin; the factory AST guard covers both new cases without native initialisation.

## Offline accounting

Tiny complete-step host/mock fixture with two valid flow rows:

| Measure | EOS/backbone | Diagonal fusion |
| --- | ---: | ---: |
| Tapes | 39 | 33 |
| Boundary tensor downloads | 100 | 86 |
| Commands | 3,980 | 3,956 |
| Conservative logical tape bytes | 145,328 | 144,880 |
| Summed planned owned tape bytes | 33,320 | 33,064 |

Each valid diagonal row replaces four tapes with one and ten boundary downloads with three. The initial forward prediction remains; loss seeds, backward output/time adjoints and target assembly intermediates stay within the graph. Parameter-gradient accumulation counts and full snapshots are unchanged. These are mock dispatch/logical storage measurements. Native latency, peak memory and complete-request performance are unmeasured.

## Verification

Evidence directory: `/workspace/exports/pocket-gpu-training-diagfused-20261002`.
Script: `/workspace/tmp/pocket-diagfused-gate.sh`.
Final gate window: **2026-10-02 05:50:46–05:51:29 UTC**.

| Gate | Passed |
| --- | ---: |
| New cases, ten repetitions | 180 top-level + 280 subtests |
| New cases, race detector | 18 top-level + 28 subtests |
| Affected host/mock regressions | 270 top-level + 401 subtests |

Final logs contain zero failures/skips. Tests cover:

- Standalone device-time flow, loss, scale and target references for times 0/.3/1, objective scales 0/.25/1 and logvar -2/0/2. Every input and named parameter VJP matches existing tolerances.
- Independent ordinary CPU flow/backward with a scalar F64 loss/seed reference; existing pinned PyTorch mixed-AD and complete-step metrics/all-gradient/update/EMA regressions remain passing.
- Zero-scale all-binding accumulation, PTX/Vulkan mock resume/replay, objective endpoints, padding and clipped multistep P/M/V/EMA.
- Complete source byte ceilings, exact-budget admission, public ownership, finite/time/scale/logvar/shape/overflow rejection, owner/scope mutation, dispatch failure, malformed and nonfinite downloaded results. Terminal retry/checkpoint/ordinary-close guards retain uncertain owners.

Affected-package vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain builds, model-layout, formatting and diff checks passed. No kernel changed; prior compiler gates were not repeated.

Scoped coverage: capability, public/private builder, result guard and shared reverse graph 100%; evaluator 88.2%, extracted loss admission 92.9%. Defensive count/alias rejection branches are partly intercepted by the resident tape's earlier validation; the evaluator's 95% helper target remains open. Whole-package and native coverage are not established.

### Development corrections and review

Initial reference tests used nonexistent model helpers/oracle fields and unprefixed CPU gradient keys; those test errors were corrected. The direct target graph was adjusted before validation to preserve the existing scale-then-subtract operation order. Malformed downloaded statistics are rejected at the resident tape boundary. Development logs are retained; no numerical tolerance changed.

The mathematical delegate timed out after 20 seconds. Independent source approval has not been obtained. Native strict-FP/FTZ, released-size resources, sustained throughput and voice quality have not been accepted; GPU and publication holds are unchanged.

## Remaining boundaries

Initial diagonal prediction, distillation flow/loss/branch handoffs, transformer/flow handoffs, forward EOS scalar, graph metadata/nonzero constants, dense embedding VJP, diagnostic synchronisation and full snapshots/live host copies remain. Overall implementation and resource/voice acceptance remain active.
