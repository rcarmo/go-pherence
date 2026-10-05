## Pocket TTS distillation composition admission

Distillation backward and forward now compose their component tapes through private helpers. Complete source/binding admission and explicit boundary, parameter and seed checks precede remapping; combined admission still precedes pruning. This closes the two older builder coverage targets without introducing test-only hooks, changing arithmetic or adding a backend scope.

The source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference is unchanged at eight paths and 469 insertions. NVIDIA/Vulkan were disabled; no native GPU, model/audio, recovery or publication run occurred.

## Composition contracts

`composeDistillTrainingTapes` admits both primary and endpoint complete sources, including dead constants and binding gradients. It retains endpoint-then-primary tensor and command order, remaps the four existing seed edges and shares live parameter identities. Only primary parameter-gradient roots are exposed. Endpoint direct parameter, condition and time gradients never accumulate. The final combined byte/graph budget includes duplicate source constants and dead placeholders.

`composeDistillForwardTapes` admits interpolation, primary and endpoint sources before joining them. The builder constructs primary then endpoint through one ordered loop, with times `{s,t}` / `{t,t}` and selected coordinates `1` / `-1`. Interpolation, advance, desired, six public outputs and ordered parameter roots keep their previous command order. New guards reject invalid delta, missing boundary positions and mismatched/duplicate parameter-name inventories.

The shared remapper also rejects nil destinations, out-of-range input/source IDs, nonconstant input placeholders and invalid shared parameter IDs. It retains extent and inventory checks. The helpers are private synchronous composition boundaries; caller source headers, constants, bindings and private edge IDs remain unchanged after success or late combined-budget rejection. Public parameter ownership and private read-only borrowing are unchanged.

## Coverage and tests

The previous combined report measured distillation builder coverage at **90.5%**, forward builder at **87.2%** and remapper at **100%** before the additional guards. Their fixed-source incompatibility branches were unexercised. The final combined gate covers every statement in both modified source files:

| Function | Final scoped coverage |
| --- | ---: |
| Distillation builder and composition helper | 100% |
| Distillation validator/evaluator and public wrapper | 100% |
| Forward builder and composition helper | 100% |
| Forward validator/evaluator and public wrapper | 100% |
| Forward remapper | 100% |

Tests pass independently valid but incompatible source graphs to the helpers. Cases reject parameter-count/name/extent mismatches, missing boundary positions, invalid seed IDs/extents and input edges, incomplete parameter-name lists, duplicate names, nonfinite dead constants, invalid remap IDs and full-source/combined byte ceilings. Source-preservation tests compare graph metadata and constant arrays before/after composition; they do not compare generated signals by hash or exact bytes.

Existing tape/independent CPU/F64/pinned PyTorch mixed-AD/minimal-LSD/FlowLM/full-step tests retain numerical tolerances. Public/private result ownership, primary-only once-per-root accumulation, objective endpoints, padding, clipping, AdamW/M/V/EMA, checkpoint/resume, cancellation and terminal-owner tests pass. Existing tiny accounting is unchanged: EOSforward still uses 11 tapes, 42 downloads, 168 downloaded elements and 3,700 commands. Native resources and latency are unmeasured.

Evidence directory: `/workspace/exports/pocket-gpu-training-composition-20261002`.
Gate script: `/workspace/tmp/pocket-composition-gate.sh`.
Final window: **2026-10-02 07:59:21--08:00:22 UTC**.

| Gate | Passed |
| --- | ---: |
| Selected new tests, ten repetitions | 90 top-level + 300 subtests |
| Selected new tests, race detector | 9 top-level + 30 subtests |
| Affected CPU/mock regressions | 432 top-level + 751 subtests |

Final logs have zero failures/skips. The selected regression coverage run covers 56.8% of Pocket TTS statements and 14.2% of pockettrain; the 100% table refers only to the two modified distillation source files. Vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain cross-builds, formatting, model-layout and diff checks pass. No kernel or replay identity changed.

### Review and limits

A mathematical judge found no obvious regression in source ordering, maps, scales or ownership, and highlighted slice aliasing, ID bounds and intermediate/final budget admission for scrutiny. Tests cover those categories. The judge did not inspect source or execute native code; independent source approval and native accuracy/resources/performance/voice acceptance are open. No numerical tolerance changed and no hold was released.

The older coverage follow-up is complete for its 95% function targets. Other internal assertions remain in the earlier report at or above the target, and native coverage is separate. Continue forward transformer/flow handoffs, prediction admission, host graph metadata/nonzero constants, dense embedding VJP, diagnostics and full snapshots/live copies under the existing holds.
