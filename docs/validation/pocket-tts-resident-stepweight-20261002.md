# Pocket TTS step reduction–weighting tape

## Implemented scope

`ptx-resident-stepweight-hybrid-v1` and `vulkan-resident-stepweight-hybrid-v1` compose step reduction with sample-weight parameter-gradient roots. Weighting consumes the reduction's scaled log-variance tensor IDs directly. The same reduction/logvar values still download for the public training-step result; they no longer upload as weighting seeds.

Training changes are uncommitted at HEAD `a90aec30567d485784f2e04d1803c543c9f2817f`. The eight-path staged inference index remains unchanged at 469 insertions. Both GPUs were disabled throughout validation. No native/model, recovery, listening or publication run occurred.

## Scaling and source admission

Step reduction applies `PEqual / validRows` to raw diagonal logvar seeds and `(1-PEqual) / validRows` to raw distillation seeds. Explicit sample-seed IDs connect those values to existing packing and MLP backward commands. Weight parameter accumulation uses unit scale and touches every named root, including zero objectives. There is no second weighting call in the new step path.

The composition admits complete reduction and weighting sources, then the combined source/bindings/byte ceiling before pruning. The source samples must contain the same finite raw seed branches as the reduction input; a mismatch is rejected. Dead raw-seed placeholders remain charged before pruning. Active masks must be a nonempty contiguous prefix, with zero derivative/statistic padding; the existing source/result guards remain in force.

All five reduction boundaries remain owned host outputs: condition gradient, direct target gradient, both scaled logvar branches and metrics. Public standalone reduction and sample-weight APIs retain their outputs and input VJPs. Public builders copy constants, while private synchronous builders borrow named parameter sources only. Capture/current capability and common owner checks guard the composed step. No arithmetic reassociation, kernel, cross-tape cache or storage reuse was added.

Post-execution malformed/nonfinite results are terminal. Retry, checkpoint boundaries and ordinary close reject uncertain owners. Weighting gradients remain resident; coherent update/snapshot ownership is unchanged.

## Offline accounting

Tiny complete-step host/mock fixture:

| Measure | Weight roots | Step/weighting |
| --- | ---: | ---: |
| Tapes | 15 | 14 |
| Boundary tensor downloads | 46 | 46 |
| Commands | 3,680 | 3,680 |
| Conservative logical tape bytes | 132,684 | 132,660 |
| Summed planned owned tape bytes | 28,472 | 28,284 |

A reduction utility tape and accumulation-only weighting tape become one bound tape. Public downloads, all parameter accumulation counts, diagnostics and full snapshots are unchanged. The scaled-seed upload is removed by device tensor edges. Native transfer cost, latency, peak memory and full-request performance are unmeasured.

## Verification

Evidence: `/workspace/exports/pocket-gpu-training-stepweight-20261002`.
Script: `/workspace/tmp/pocket-stepweight-gate.sh`.
Final window: **2026-10-02 06:41:40–06:42:24 UTC**.

| Gate | Passed |
| --- | ---: |
| New tests, ten repetitions | 170 top-level + 310 subtests |
| New tests, race detector | 17 top-level + 31 subtests |
| Affected host/mock regressions | 368 top-level + 596 subtests |

Final logs contain zero failures/skips. Tests compare reduction plus standalone seeded weighting at 1/3/17 frames, partial padding and objective weights 0/.25/1. Independent scalar F64 affine/ReLU parameter VJPs agree with the graph. PTX/Vulkan mock driver captures every accumulated parameter gradient and compares it with the composed source; accumulation counts remain once per root and unit-scaled.

Existing independent F64 step-reduction and weighting references, pinned PyTorch full-step/mixedAD/minimal-LSD, objective endpoint, padding, clipping, P/M/V/EMA and checkpoint/resume regressions pass unchanged tolerances. Tests cover combined exact/undersized budgets, sample/raw-seed mismatch, malformed active masks, public ownership/private parameter-only borrowing, dead-invalid constants and result finite/shape/padding/sample-alias guards. Scope/owner mutation, dispatch and post-execution failure tests block retry/checkpoint and retain owners.

Affected-package vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain builds, formatting, model-layout and diff checks passed. No kernel changed; compiler gates were not repeated. Scoped capability/public wrapper/decoder coverage is 100%, builder 95.7%, evaluator 93.3%. Earlier resident validation intercepts some defensive evaluator branches; its 95% helper target stays open. Whole-package/native coverage is not established.

A mathematical judge confirmed the stated once-only scaling and observational host-output contract, provided valid-row count is guarded and packing only reindexes seed values. Source admission enforces a nonzero active prefix. The judge did not inspect source or execute native code. Independent source approval, native strict-FP/FTZ, released-size resources, sustained throughput and voice quality have not been accepted. Holds are unchanged.

## Remaining boundaries

Prediction/admission and transformer/flow handoffs, forward EOS scalar, graph metadata/nonzero constants, dense embedding VJP, diagnostic synchronisation and full snapshots/live host copies remain. Overall implementation and native/resource/voice acceptance remain active.
