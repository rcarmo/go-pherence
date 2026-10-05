# Pocket TTS private final-step result selection

## Implemented scope

`ptx-resident-stepresults-hybrid-v1` and `vulkan-resident-stepresults-hybrid-v1` omit two discarded full arrays from the composed final-step downloads: reduced condition gradient and direct target gradient. The private step returns scaled logvar branches, metrics, optional voice gradient, final normalized gradient and a direct-target last-row witness. Public component APIs retain their full outputs.

Training changes are uncommitted at HEAD `a90aec30567d485784f2e04d1803c543c9f2817f`. The staged inference index remains at eight paths and 469 insertions. Both GPUs were disabled during validation. No native/model, recovery, listening or publication run occurred.

## Result contract

Condition/direct-target tensors still execute as dependencies of backbone seed generation and final input addition. Only their full host materialisations disappear. The private result has nil condition/direct-target fields rather than misleading zero-filled full outputs. The step does not consume those fields and uses its validated logvar/metrics/input results instead. Public reduction, weighting and backbone builders and APIs keep existing shapes and VJPs.

A new bounded slice downloads the direct target's final row. The decoder compares it exactly with the final normalized gradient row, preserving the existing contract that the conditioner normalized-row VJP is zero. Both values use the same F32 graph path with no recast/repack. Padding also requires zero logvar outputs and a zero final direct-tail witness when the last row is inactive.

Every selected output must have the expected length, be finite, disjoint from other outputs and independent of caller arrays. Metrics retain finite/nonnegative square/EOS checks. Complete source/binding/byte admission runs before result selection and again after adding the tail root; dead-invalid constants cannot hide behind pruning. Parameter-gradient roots, scale and accumulation counts are unchanged. Failed output or tail validation is terminal and blocks retry/checkpoint/ordinary close while retaining owners.

## Offline accounting

Tiny complete-step host/mock fixture:

| Measure | Step/backbone | Selected results |
| --- | ---: | ---: |
| Tapes | 12 | 12 |
| Boundary tensor downloads | 45 | 44 |
| Downloaded result/gradient elements | 190 | 174 |
| Commands | 3,682 | 3,683 |
| Conservative logical tape bytes | 132,744 | 132,756 |
| Summed planned owned tape bytes | 27,952 | 27,960 |

Two full array downloads are replaced by one latent-width tail witness. The extra command and bounded tail storage preserve the exact guard. Full snapshots and diagnostic transfers are unchanged; the element counter covers tape downloads only. Native latency, peak memory and complete-request performance are unmeasured.

## Verification

Evidence: `/workspace/exports/pocket-gpu-training-stepresults-20261002`.
Script: `/workspace/tmp/pocket-stepresults-gate.sh`.
Final window: **2026-10-02 06:58:06–06:58:51 UTC**.

| Gate | Passed |
| --- | ---: |
| New tests, ten repetitions | 170 top-level + 170 subtests |
| New tests, race detector | 17 top-level + 17 subtests |
| Affected host/mock regressions | 394 top-level + 643 subtests |

Final logs contain zero failures/skips. Cases compare selected results with the full composed tape for empty/nonempty voice, all returned seeds/gradients, exact/full-source byte ceilings and malformed/dead-invalid sources. Output tests reject wrong counts/shapes, nonfinite values, negative statistics, inactive logvars, aliases/borrowed arrays, changed tails and nonzero inactive direct tails.

PTX/Vulkan mock full-step objective endpoints, padding, clipped multistep P/M/V/EMA, checkpoint/resume, unchanged parameter accumulation and public reduction-output contracts pass. Post-execution result/tail rejection blocks retry/checkpoint/ordinary close. Existing independent CPU/F64, pinned PyTorch mixedAD/minimal-LSD/complete-step regressions pass unchanged tolerances.

Affected-package vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain builds, formatting, model-layout and diff checks passed. No kernel changed; compiler gates were not repeated. All four functions in `training_step_results.go` have 100% scoped statement coverage. Whole-package/native coverage is not established.

### Development correction and review

An initial test-file write accidentally included shell text, which was removed before compilation; a duplicate failure-test name was renamed. The first compiled run and final gates are retained. Numerical tolerances are unchanged.

A mathematical judge confirmed that removing pure host materialisations preserves the stated roots/scaling, with the caveat that exact tail equality must use the same precision/layout. This implementation slices the existing direct tensor and does not recast it. The judge did not inspect source or execute native code. Independent source approval, native strict-FP/FTZ, released-size resources, sustained throughput and voice quality have not been accepted. Holds are unchanged.

## Remaining boundaries

Prediction/admission values, forward transformer/flow handoffs, forward EOS scalar, graph metadata/nonzero constants, dense embedding VJP, diagnostic synchronisation and full snapshots/live host copies remain. Overall implementation and native/resource/voice acceptance remain active.
