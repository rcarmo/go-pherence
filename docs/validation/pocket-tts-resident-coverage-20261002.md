## Pocket TTS fused-boundary failure coverage

Four fused component evaluators now use private decoders for their existing public results. The decoders centralise count, shape, finite-value and ownership checks without changing graph construction, arithmetic, scaling, result ownership or replay scopes. New tests exercise malformed decoder inputs and post-download aliases/borrowed results, then check terminal owner retention.

The work is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference is unchanged: eight paths, 469 insertions. NVIDIA/Vulkan were disabled; no native GPU, model/audio, recovery or publication run occurred.

## Scoped coverage

The baseline combines component tests and resident mock tests, avoiding the misleading zeros in a single-slice coverage profile. The final gate also includes the affected reference regressions. The selectors are recorded in `commands.sh`; baseline commands are in `baseline-commands.txt`.

| Function | Baseline | Final |
| --- | ---: | ---: |
| Backbone builder | 92.0% | 98.0% |
| Backbone evaluator | 97.0% | 100.0% |
| Diagonal evaluator | 88.2% | 100.0% |
| Primary evaluator | 89.5% | 100.0% |
| Distillation evaluator | 89.5% | 100.0% |
| Distillation-forward evaluator | 88.2% | 100.0% |
| Forward composition remap | 90.9% | 100.0% |
| Stepweight evaluator | 93.3% | 100.0% |
| Stepbackbone evaluator | 96.2% | 96.2% |
| Stepbackbone decoder | 97.0% | 100.0% |

All four new decoders have **100% scoped statement coverage**. Diagonal and primary builders/validators are 100%. Stepweight builder is 95.7%; stepbackbone builder is 98.4%. The final selected tests cover 56.6% of the Pocket TTS package and 14.2% of pockettrain. These package percentages include unrelated and native code; the function table has a narrower scope.

### Builder targets at this gate

Follow-up: [composition admission validation](pocket-tts-resident-composition-20261002.md) closes these two builder targets at 100% scoped coverage. The figures below preserve the earlier gate state.

Distillation builder measured **90.5%** and distillation-forward builder at **87.2%**, below the 95% target. Fixed internal builders make several source/seed/inventory consistency guards difficult to trigger through public callers. Those targets are not complete and have no waiver.

The distillation gaps check endpoint complete bindings, identical primary/endpoint inventories and parameter extents, and four explicit seed IDs/extents. Forward gaps check the second flow builder, independent complete binding admission of each fixed graph, and internal composition errors. Standalone remap tests now cover input extent, missing parameter inventory and mismatched parameter extent. The remaining source-builder paths need a bounded composition seam or separate refactor; they are retained without adding test-only hooks or weakening admission.

Backbone's remaining 2% checks final resident-binding validation after the same materialised tape has already passed it. Stepweight's 4.3% checks fixed weight bindings and seed extents. Stepbackbone's 1.6% checks duplicate parameter names between distinct fixed namespaces. Its evaluator has one selected-result augmentation failure branch still uncovered in this combined report. These assertions remain in source.

## Tests and retention

Direct decoder tests reject wrong counts, statistics shapes, NaN, negative square statistics, overlapping outputs and caller-backed outputs. Mock evaluator tests cover diagonal, primary, distillation, distillation-forward, backbone, stepweight and stepbackbone ownership failures. Shapes, finiteness and inactive-row zeros are preserved in the injected ownership faults, so the decoder itself rejects them. Each rejection is terminal and blocks another prepared step, coherent boundary validation and ordinary close.

Internal backbone composition tests reject an extra parameter without a gradient, mismatched parameter/gradient extents and more than 4096 tensors. The fixed graph ceilings and complete-source budgets are unchanged. No generated-signal equality or tolerance widening was introduced.

Evidence directory: `/workspace/exports/pocket-gpu-training-coverage-20261002`.
Gate script: `/workspace/tmp/pocket-coverage-gate.sh`.
Final window: **2026-10-02 07:51:01--07:51:46 UTC**.

| Gate | Passed |
| --- | ---: |
| New tests, ten repetitions | 90 top-level + 230 subtests |
| New tests, race detector | 9 top-level + 23 subtests |
| Affected CPU/mock regressions | 428 top-level + 724 subtests |

Final logs have zero failures/skips. Independent CPU/F64 and pinned PyTorch mixed-AD/FlowLM/full-step regressions pass at unchanged tolerances. Vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain cross-builds, formatting, model-layout and diff checks pass. This slice makes no performance claim and changes no kernel or backend scope.

### Development and review

A borrowed-output test initially used a separately allocated condition array, so it correctly failed to trigger ownership rejection. Stepweight borrowed-sample tests initially violated inactive-row zeros before reaching the ownership check; the injected arrays now preserve those values. A test expected the wrong binding-error wording. Gate-script generation also failed twice before the final corrected script ran. Retained logs distinguish development failures from the successful final gate. Numerical thresholds did not change.

A delegated read-only branch review timed out after 60 seconds and supplied no approval. Independent source review, native accuracy/resources/performance and voice/listening acceptance are open. Execution/publication holds are unchanged.

## Next work

The subsequent composition-admission gate closed the two remaining builder targets. Continue forward transformer/flow handoffs and prediction admission. Host metadata/nonzero constants, dense embedding VJP, diagnostics and full snapshots/live copies still require implementation work. The overall goal stays active.
