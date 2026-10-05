# Pocket TTS combined distillation forward tape

## Implemented scope

`ptx-resident-forwardfused-hybrid-v1` and `vulkan-resident-forwardfused-hybrid-v1` compose interpolation, primary flow/selected-time JVP, advance and endpoint flow in one resident-bound tape. Neither forward graph accumulates parameter gradients. Six owned outputs preserve the values needed by the public distillation result and existing host loss-overflow admission.

Training changes are uncommitted at HEAD `a90aec30567d485784f2e04d1803c543c9f2817f`. The eight-path staged inference index remains at 469 insertions. Both GPUs were disabled during validation. No model/native GPU, recovery, listening or publication run occurred.

## Forward contract

The tape preserves the existing operation order:

- `xs = s * target + (1-s) * noise`, with separate multiply/add primitives.
- Evaluate primary flow at `[s,t]` using `xs`, with selected-time JVP index 1.
- `xt = xs + (t-s) * velocity`; `desired = velocity + (t-s) * tangent`.
- Evaluate endpoint flow at `[t,t]` using `xt`, with no selected tangent.

Explicit flow-input tensor IDs connect interpolation and advance to each flow graph. Both use the same read-only live parameter IDs. The public output order is `xs`, velocity, tangent, `xt`, desired and endpoint prediction. Output guards require finite, correctly sized, disjoint, caller-independent arrays.

Both complete flow source graphs and interpolation pass admission before composition. The final graph checks the combined source, parameter bindings and byte ceiling before pruning. Dead zero/parameter placeholders and unused interpolation desired output remain charged before pruning. Public builders own constants; private builders borrow named parameter sources only. No cross-tape cache or storage reuse was introduced.

Execution sets `accumulate=false`. Gradient bindings exist for admission only; they are not roots of the pruned forward execution. The existing fused backward retains primary-only parameter accumulation and endpoint input-only stop-gradient behavior. Older scopes retain separate interpolation/flow/advance/endpoint calls. The initial diagonal forward and loss admission are unchanged.

## Offline accounting

Tiny complete-step host/mock fixture, two valid flow rows:

| Measure | Distillation backward fusion | Forward fusion |
| --- | ---: | ---: |
| Tapes | 23 | 17 |
| Boundary tensor downloads | 54 | 50 |
| Commands | 3,694 | 3,688 |
| Conservative logical tape bytes | 136,168 | 132,952 |
| Summed planned owned tape bytes | 29,456 | 28,536 |

Each valid row replaces four tapes/eight downloads with one tape/six downloads. The unused interpolation output and endpoint tangent no longer download. The six required public/admission boundaries remain. Parameter accumulation counts, diagnostics and full snapshots are unchanged. Native latency, peak memory and complete-request performance are unmeasured.

## Verification

Evidence: `/workspace/exports/pocket-gpu-training-forwardfused-20261002`.
Script: `/workspace/tmp/pocket-forward-gate.sh`.
Final gate window: **2026-10-02 06:19:18–06:20:11 UTC**.

| Gate | Passed |
| --- | ---: |
| New tests, ten repetitions | 180 top-level + 260 subtests |
| New tests, race detector | 18 top-level + 26 subtests |
| Affected host/mock regressions | 327 top-level + 516 subtests |

Final logs contain zero failures/skips. References compare all six outputs with standalone interpolation/device-time primary flow/JVP/advance/endpoint tapes for s=0/.3/1 and t=s/1, and independently with ordinary CPU forward/JVP. Existing pinned PyTorch minimal-LSD/mixedAD/complete-step regressions pass without tolerance changes.

PTX/Vulkan mock tests prove no forward parameter accumulation, six boundary roots and unchanged all-gradient/update/EMA, objective endpoint, padding, clipping and checkpoint/resume behavior. Tests cover combined exact/undersized budgets, invalid topology/time/source shapes/values, public ownership/private parameter-only borrowing, dead-invalid source admission, nonfinite/alias/output-shape guards, scope/owner changes and dispatch/post-execution failure retention.

Affected-package vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain builds, formatting, model-layout and diff checks passed. No kernel changed; compiler gates were not repeated. Scoped capability/public wrapper/result guard coverage is 100%, builder 89.0%, evaluator 88.2%. Some defensive remap/inventory/extent and malformed driver-return branches were not reached; their 95% helper targets stay open. Whole-package/native coverage is not established.

### Development correction and review

Initial mock accounting expected one saved download per valid row. It measured two: the unused interpolation output and endpoint tangent. The expectations now reflect those eliminated boundaries. Numerical tolerances are unchanged; the initial failure log is retained.

A mathematical judge confirmed forward equivalence under the stated read-only parameter, explicit-input-edge and no-accumulation invariants. It did not inspect source or test native execution. Independent source approval, native strict-FP/FTZ, released-size resources, sustained performance and voice quality have not been accepted. GPU and publication holds are unchanged.

## Remaining boundaries

Six forward prediction/admission values, initial diagonal prediction, transformer/flow handoffs, forward EOS scalar, metadata/nonzero constants, dense embedding VJP, diagnostic synchronisation and full snapshots/live host copies remain. Overall implementation and native/resource/voice acceptance remain active.
