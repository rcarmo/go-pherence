## Pocket TTS combined backbone/EOS forward

`ptx-resident-eosforward-hybrid-v1` and `vulkan-resident-eosforward-hybrid-v1` execute the training step's read-only backbone and EOS mean in one tape. Sequence and transformed rows keep their existing owned host boundaries. The scalar EOS result also returns to host. Public FlowLM APIs and earlier scopes keep their previous dispatch.

The source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. The staged inference index contains eight paths and 469 insertions, unchanged. NVIDIA/Vulkan were disabled during validation. No released-model/audio, native GPU, recovery or publication run occurred.

## Admission and arithmetic

The private builder admits the complete EOS/backbone backward source with zero seeds and shape-compatible hidden placeholders. It adds unweighted BCE rows/mean using the existing EOS affine input edge, retains every parameter/gradient binding and validates the combined logical budget before dispatch. Dependency pruning executes only sequence, transformed rows and the scalar mean; no parameter gradients accumulate.

The host applies the same EOS absolute-dot overflow bound to the actual downloaded audio hidden rows. This check occurs after the read-only forward and before any backward/accumulation/update. A cancelled sum with finite logits still fails if its absolute-dot bound exceeds the existing ceiling. Padding rows remain part of that bound. Invalid count/shape/nonfinite/aliased/borrowed outputs or negative loss fail terminally; retry and checkpoint are blocked, and uncertain owners remain reachable.

Scalar loss is unweighted, as before. The unchanged EOS/backbone backward applies the configured EOS weight once, joins flow seeds and accumulates parameters once. Objective weights, first-invalid-frame labels, padding and empty/nonempty voice behaviour are unchanged. The private tape stores the same mask/weight and backend owners required by backward. Public forward/backward continues returning sequence, hidden rows and EOS logits.

Read-only EOS execution now precedes its data-dependent host bound. Rejected samples may perform extra forward work, but no gradients or optimiser state mutate. Native floating-point exceptions/precision and allocator behaviour still need qualification.

## Tiny complete-step accounting

| Measure | Distillresults | EOSforward |
| --- | ---: | ---: |
| Tapes | 12 | 11 |
| Boundary tensor downloads | 42 | 42 |
| Downloaded result/gradient elements | 168 | 168 |
| Commands | 3,697 | 3,700 |
| Conservative logical tape bytes | 132,908 | 133,088 |
| Summed planned owned tape bytes | 27,960 | 27,864 |

The separate read-only EOS tape disappears. Accumulation counts and full snapshot/diagnostic transfers are unchanged. The combined source adds three executed commands and 180 summed logical bytes; planned owned storage falls by 96 bytes in this fixture. There is no measured native latency or released-size memory improvement.

## Offline verification

Evidence directory: `/workspace/exports/pocket-gpu-training-eosforward-20261002`.
Gate script: `/workspace/tmp/pocket-eosforward-gate.sh`.
Final window: **2026-10-02 07:37:38--07:38:25 UTC**.

| Gate | Passed |
| --- | ---: |
| New tests, ten repetitions | 170 top-level + 270 subtests |
| New tests, race detector | 17 top-level + 27 subtests |
| Affected CPU/mock regressions | 422 top-level + 701 subtests |

Final logs contain zero failures/skips. Tape composition tests compare sequence/transformed rows and the EOS scalar against separate backbone/EOS tapes for 1/2/7 frames, empty/nonempty voice and weights 0/.1/1/2. They check unchanged parameter identity, complete combined source budgets and invalid dead sources.

Host/mock tests cover read-only accumulation counts, public full FlowLM outputs, complete-step gradients/P/M/V/EMA, objective endpoints, padding, clipped multistep updates, checkpoint/resume and cross-scope rejection. Post-download overflow rejection leaves accumulation counts unchanged. Shape/nonfinite/negative/alias/source/owner/scope/mask/budget/empty-text failures and every tape construction failure block retry/checkpoint and ordinary close. Existing independent F64/CPU and pinned PyTorch transformer/FlowLM/mixed-AD/full-step regressions pass unchanged tolerances.

Scoped statement coverage: enable/evaluator/decoder **100%**, builder **96.2%**. The unexercised builder guard detects a missing EOS command after the fixed internal EOS/backbone builder succeeds; that builder always inserts the edge. It is retained as an internal topology assertion. Whole-package/native coverage is not established. Vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain cross-builds, formatting, layout and diff checks pass. No kernel changed.

### Development and review

Development logs retain a notice-contract failure (missing the literal CPU label), fixture helper/type compile mistakes, and their corrections. Numerical tolerances did not change.

A mathematical judge confirmed that moving pure read-only EOS forward before the same host bound does not change accepted updates, provided that bound gates all backward/accumulation. It did not inspect source or run native code. Independent source approval, native accuracy/resources/throughput and new-voice quality are open. Holds are unchanged.

## Next boundaries

Forward transformer-to-flow handoffs and prediction admission, host graph metadata/nonzero constants, dense embedding VJP, pre-update diagnostics and full snapshots/live host copies still require work. EOS mean still transfers as a host scalar; this slice removes one tape, without a full-device scope. Older builder/evaluator coverage gaps stay tracked separately.
