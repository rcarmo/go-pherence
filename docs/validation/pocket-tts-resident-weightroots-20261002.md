# Pocket TTS step-only weighting gradient roots

## Implemented scope

`ptx-resident-weightroots-hybrid-v1` and `vulkan-resident-weightroots-hybrid-v1` use a private accumulation-only sample-weight backward path in the training step. The step discards the MLP's repeated forward output and input VJP; neither value is requested from the device. Public seeded weighting APIs still return both real outputs.

Training changes are uncommitted at HEAD `a90aec30567d485784f2e04d1803c543c9f2817f`. The staged inference index remains unchanged: eight paths and 469 insertions. Both GPUs were disabled throughout validation. No native/model, recovery, listening or publication run occurred.

## Root selection and safety

The private path builds the same complete sample-weight tape with already-scaled diagonal/distillation logvar seeds. Source shape/finiteness, sample lengths, model topology, bindings and conservative byte ceiling are admitted before pruning. It requests no boundary tensor downloads and accumulates every named parameter-gradient root with unit scale. Zero seeds still touch all parameter bindings.

Pruning can remove the final MLP input-VJP operations and repeated forward-output root where no accumulated gradient needs them. It preserves forward activations needed by parameter gradients. Frozen sample times receive no derivatives from the training step. Legacy scopes retain their seeded call, full outputs and input VJP; the public `EvaluateSampleWeight` and builder contracts are unchanged in the new scope as well.

The step captures the private capability before forward and rejects its removal/disablement before backward. The resident backend checks current model/weight owner and scope. Active tape borrow, parameter-only private source borrowing, logical ceilings, error handling and uncertain-owner retention are unchanged. There is no new kernel, arithmetic reassociation, cross-tape cache or storage reuse.

The resident runner still calls the driver's output-completion method with an empty list. It requests zero tensor downloads. Execution, accumulation and close errors remain visible even without returned boundary values; uncertain close makes the backend terminal and retains the owner.

## Offline accounting

Tiny complete-step host/mock fixture:

| Measure | Diagonal forward | Weight roots |
| --- | ---: | ---: |
| Tapes | 15 | 15 |
| Boundary tensor downloads | 48 | 46 |
| Commands | 3,684 | 3,680 |
| Conservative logical tape bytes | 132,904 | 132,684 |
| Summed planned owned tape bytes | 28,472 | 28,472 |

All parameter accumulation counts, updates, diagnostic synchronisation and full snapshots are unchanged. Native latency, peak memory and complete-request performance are unmeasured. The empty-list completion call has not been qualified on native PTX/Vulkan drivers.

## Verification

Evidence: `/workspace/exports/pocket-gpu-training-weightroots-20261002`.
Script: `/workspace/tmp/pocket-weightroots-gate.sh`.
Final gate window: **2026-10-02 06:33:16–06:34:05 UTC**.

| Gate | Passed |
| --- | ---: |
| New tests, ten repetitions | 140 top-level + 230 subtests |
| New tests, race detector | 14 top-level + 23 subtests |
| Affected host/mock regressions | 353 top-level + 565 subtests |

Final logs contain zero failures/skips. PTX/Vulkan mock tests compare captured accumulation with the public tape and independent scalar F64 affine/ReLU parameter VJPs for zero and nonzero seeds. Repeated calls add one contribution each. Every gradient root is touched; no boundary tensor downloads occur. Public seeded/non-accumulating output and input VJP and ownership remain intact.

Full-step objective endpoints, padding, clipping, P/M/V/EMA and checkpoint/resume pass against existing pinned PyTorch and independent CPU references with unchanged tolerances. Source/seed/frame/byte/scope/owner admission, dispatch failure, retry/checkpoint blocking and uncertain close retention are covered. Existing sample-weight reference tests cover larger frame counts and ReLU boundaries in the regression gate.

Affected-package vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain builds, formatting, model-layout and diff checks passed. No kernel changed; compiler gates were not repeated. Both functions in `training_weight_roots.go` have 100% statement coverage in the scoped new-case run. Whole-package/native coverage is not established.

A mathematical judge confirmed that dropping only discarded boundary roots preserves the stated parameter gradients and unit scaling when their seeds and accumulation arithmetic remain unchanged. It did not inspect source or execute native code. Independent source approval, native strict-FP/FTZ, released-size resources, sustained throughput and voice quality have not been accepted. GPU and publication holds are unchanged.

## Remaining boundaries

Prediction/admission values, transformer/flow handoffs, forward EOS scalar, graph metadata/nonzero constants, dense embedding VJP, diagnostic synchronisation and full snapshots/live host copies remain. Overall implementation and native/resource/voice acceptance remain active.
