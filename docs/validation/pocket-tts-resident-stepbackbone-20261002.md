# Pocket TTS step reduction–backbone backward tape

## Implemented scope

`ptx-resident-stepbackbone-hybrid-v1` and `vulkan-resident-stepbackbone-hybrid-v1` compose step reduction/weighting, EOS/backbone backward and final direct-target gradient addition in one resident-bound tape. Reduction condition seeds feed a zero-prefix padded transformer seed through tensor IDs. The final normalized-latent gradient adds the direct target term in graph.

Training changes are uncommitted at HEAD `a90aec30567d485784f2e04d1803c543c9f2817f`. The staged inference index remains unchanged: eight paths and 469 insertions. Both GPUs were disabled during validation. No native/model, recovery, listening or publication run occurred.

## Gradient and ownership contract

The existing reduction computes combined condition/direct-target gradients and scales raw logvar branches once. Weighting uses those seed IDs with unit accumulation. The reduction condition output is padded with zero prefix rows and replaces the explicit backbone seed constant. EOS VJP retains its single objective-weight factor; backbone and weighting parameter names are disjoint and all bindings accumulate once at unit scale.

Raw normalized conditioner VJP plus reduced direct target uses the existing final-add order. The final normalized row of the conditioner VJP is zero; after addition, that row equals the direct-target row. Caller-owned normalized latents are the training target space, so the direct term needs no additional normalization derivative. Endpoint input-only and primary mixed-AD stop-gradient policies are unchanged.

The step bypasses separate weighting/backbone/final-add execution only under the captured new capability and unchanged EOS/backbone owner/mask/weight state. Public reduction outputs, logvar branches, metrics and final raw input gradients remain owned. Standalone reduction/weighting/EOS/backbone APIs and older scopes retain their outputs and execution order.

Complete reduction/weighting and backbone source graphs are admitted before composition, followed by combined source/binding/byte admission before pruning. Dead placeholders and source constants remain charged; the 4,096 tensor/command ceiling applies to the combined graph. Public builders own constants, private named parameters are read-only borrowed synchronously, and no cross-tape cache or storage reuse is introduced. Output finite/shape/alias/final-row guards and terminal retry/checkpoint/ordinary-close retention remain enforced.

## Offline accounting

Tiny complete-step host/mock fixture:

| Measure | Step/weighting | Step/backbone |
| --- | ---: | ---: |
| Tapes | 14 | 12 |
| Boundary tensor downloads | 46 | 45 |
| Commands | 3,680 | 3,682 |
| Conservative logical tape bytes | 132,660 | 132,744 |
| Summed planned owned tape bytes | 28,284 | 27,952 |

The composed tape retains five public reduction boundaries plus optional voice VJP and final normalized gradient. The intermediate normalized conditioner gradient no longer downloads; reduction seeds and direct target are not re-uploaded to downstream tapes. Commands/logical bytes increase slightly due explicit prefix padding. Parameter accumulation counts, diagnostics and full snapshots are unchanged. Native latency, peak memory and complete-request performance are unmeasured.

## Verification

Evidence: `/workspace/exports/pocket-gpu-training-stepbackbone-20261002`.
Script: `/workspace/tmp/pocket-stepbackbone-gate.sh`.
Final window: **2026-10-02 06:49:25–06:50:17 UTC**.

| Gate | Passed |
| --- | ---: |
| New tests, ten repetitions | 170 top-level + 300 subtests |
| New tests, race detector | 17 top-level + 30 subtests |
| Affected host/mock regressions | 381 top-level + 626 subtests |

Final logs contain zero failures/skips. References compare standalone step/weighting plus EOS/backbone and independent final-add arithmetic for empty/nonempty voice, repeated text, padding and weights 0/.25/1. Every parameter gradient and public reduction/input result matches existing tolerances. PTX/Vulkan mock captures all accumulated gradients once; count and numerical values match the composed source.

Pinned PyTorch complete-step/all-gradient/update/EMA and independent CPU/F64 reduction/weighting/backbone regressions pass, including objective endpoints, clipped multistep P/M/V/EMA and checkpoint/resume. Tests cover combined exact/undersized budgets, topology/row/input/hidden/source mismatch, public ownership, output count/shape/nonfinite/overlap/source aliases/final-row rejection and scope/owner mutation. Dispatch/post-execution failure blocks retry/checkpoint/ordinary close and retains owners.

Affected-package vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain builds, formatting, model-layout and diff checks passed. No kernel changed; compiler gates were not repeated. Scoped capability/public wrapper coverage is 100%, builder 98.4%, decoder 97.0%, evaluator 94.4%. The evaluator's 95% helper target stays open for defensive malformed return handling intercepted by earlier resident validation. Whole-package/native coverage is not established.

A mathematical judge found no once-scaling issue under the stated disjoint parameter and unchanged-operation assumptions. Its caveat concerns higher-order autodiff through seed edges: this implementation explicitly connects numerical VJP seeds, without an automatic reverse pass through those seed-generation nodes. Mixed-AD flow operations are unchanged. The judge did not inspect source or execute native code. Independent source approval, native strict-FP/FTZ, released-size resources, sustained throughput and voice quality have not been accepted. Holds are unchanged.

## Remaining boundaries

Prediction/admission values, forward transformer/flow handoffs, forward EOS scalar, graph metadata/nonzero constants, dense embedding VJP, diagnostic synchronisation and full snapshots/live host copies remain. Overall implementation and native/resource/voice acceptance remain active.
