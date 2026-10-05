# Pocket TTS combined distillation backward tape

## Implemented scope

`ptx-resident-distillfused-hybrid-v1` and `vulkan-resident-distillfused-hybrid-v1` compose endpoint loss/input-only VJP and primary mixed backward/finish in one resident-bound tape. Mixed value/tangent seeds, endpoint input derivative and delta derivative connect through tensor IDs. Only primary parameter gradients accumulate, scaled once by the objective weight.

Training changes are uncommitted at HEAD `a90aec30567d485784f2e04d1803c543c9f2817f`. The eight-path staged inference index remains at 469 insertions. Both GPUs were disabled during validation. No native/model, recovery, listening or publication run occurred.

## Composition and admission

The existing endpoint and primary builders admit their full source graphs. Composition replaces explicit primary seed-input edges with endpoint outputs. Parameter IDs in both graphs point to the same read-only live resident parameters. The exposed gradient binding for each name points exclusively to its primary gradient. Endpoint parameter-gradient commands remain in the admitted source but are not accumulation roots; pruning drops them. No automatic reverse pass differentiates through seed-generation nodes.

Raw condition/target/noise/time derivatives and loss/logvar statistics are unchanged. Later step reduction applies raw distillation objective scaling once; resident parameter accumulation uses the caller scale once. The endpoint parameter/condition/time stop-gradient policy is preserved, including nonzero and zero objective weights.

The combined logical ceiling includes both source graphs, dead placeholders and duplicate parameter sources before pruning. Source shape/finiteness, parameter inventory/extents, edge extents and complete binding admission run before device execution. Both builders retain public owned constants and private synchronous named-parameter-only borrowing. No cross-tape cache or storage reuse was introduced. Legacy scopes retain separate endpoint and primary calls.

Forward primary/endpoint values still cross host boundaries and feed the existing conservative loss overflow gate. The public minimal-LSD velocity/tangent/endpoint/residual fields remain available.

## Offline accounting

Tiny complete-step host/mock fixture, two valid flow rows:

| Measure | Primary fusion | Combined distillation |
| --- | ---: | ---: |
| Tapes | 25 | 23 |
| Boundary tensor downloads | 62 | 54 |
| Commands | 3,694 | 3,694 |
| Conservative logical tape bytes | 139,296 | 136,168 |
| Summed planned owned tape bytes | 31,536 | 29,456 |

Each valid row replaces two tapes/nine downloads with one tape/five downloads. Raw derivatives and statistics are returned, while intermediate endpoint input/mixed seeds/delta stay in graph. Parameter accumulation counts and full snapshots are unchanged. Native latency, peak memory and complete-request resource use are unmeasured.

## Verification

Evidence: `/workspace/exports/pocket-gpu-training-distillfused-20261002`.
Script: `/workspace/tmp/pocket-distill-gate.sh`.
Final window: **2026-10-02 06:12:19–06:13:06 UTC**.

| Gate | Passed |
| --- | ---: |
| New tests, ten repetitions | 180 top-level + 280 subtests |
| New tests, race detector | 18 top-level + 28 subtests |
| Affected host/mock regressions | 313 top-level + 490 subtests |

Final logs contain zero failures/skips. References compare standalone endpoint plus primary tapes for s=0/.3/1 and t=s/1, every raw derivative/statistic and every named primary parameter gradient. Binding tests require the endpoint parameter identity and the primary gradient ID; endpoint gradients are not included. Independent ordinary CPU endpoint VJP and CPU mixed-AD plus F64 loss references pass existing tolerances.

PTX/Vulkan mock full-step objective endpoints, padding, clipping, P/M/V/EMA and checkpoint/resume pass. Driver-captured parameter gradients equal primary raw gradients times scale 0/.25/1; raw input derivatives are scale-independent. Exactly one contribution per primary parameter is captured. Existing pinned PyTorch minimal-LSD/mixedAD/complete-step regressions pass without tolerance changes.

Tests cover combined exact/undersized budgets, invalid source/time/logvar/shape/overflow inputs, public ownership/private borrowing, dead-invalid constants, result shape/nonfinite/alias rejection, scope/owner changes, dispatch and post-execution failure retention. Failed work cannot retry, checkpoint or ordinarily close its uncertain owner.

Affected-package vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain builds, formatting, model-layout and diff checks passed. No kernel changed; compiler gates were not repeated. Scoped capability/public wrapper/result guard coverage is 100%, builder 90.5%, evaluator 89.5%. Defensive impossible-after-valid-source inventory/edge checks and some early resident return guards are not reached; the 95% helper targets stay open. Whole-package/native coverage is not established.

The mathematical delegate timed out after 10 seconds. Independent source approval has not been obtained. Native strict-FP/FTZ, released-size resources, sustained performance and voice quality have not been accepted. GPU and publication holds are unchanged.

## Remaining boundaries

Primary/endpoint forward prediction admission, initial diagonal prediction, transformer/flow handoffs, forward EOS scalar, metadata/nonzero constants, dense embedding VJP, diagnostic synchronisation and full snapshots/live host copies remain. Overall implementation and native/resource/voice acceptance remain active.
