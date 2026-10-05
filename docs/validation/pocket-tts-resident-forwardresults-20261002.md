## Pocket TTS selected forward results

`ptx-resident-forwardresults-hybrid-v1` and `vulkan-resident-forwardresults-hybrid-v1` download only audio hidden rows, the unweighted EOS mean and a scalar finite witness from the training step's read-only forward tape. The full sequence and transformed prefix stay inside the graph. Public FlowLM APIs and older scopes retain their full outputs.

The source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference is unchanged: eight paths and 469 insertions. NVIDIA/Vulkan were disabled during validation. No native GPU, model/audio, recovery or publication run occurred.

## Private result contract

The existing EOS/backbone source is admitted in full before selection. All parameter bindings, gradient outputs, graph ceilings and conservative logical budgets remain. The private builder adds an audio-hidden slice and runtime finite witness, then admits the augmented graph before pruning.

The composed step backward reevaluates the immutable backbone from the original batch and never consumes the downloaded sequence or transformed prefix. Private omitted `sequence`, `output.Sequence` and `output.EOS` fields are nil, without fabricated full arrays. Returned audio hidden rows are owned and disjoint from the scalar outputs and caller arrays. The captured mask, weight and backend owners retain the existing backward contract. Public `ForwardBackward` still returns sequence and EOS logits.

The finite witness covers the full original sequence and transformed rows, including the omitted prefix. Runtime zero-weight linear products propagate NaN/Inf while finite maximum values produce zero without magnitude-sum overflow. The existing strict-FP kernels load zero weights as tensor inputs; no new kernel or constant-folding optimisation was introduced. Witness chunks are at most 16,384 elements, matching `LinearShape`'s input-width ceiling. This avoids rejecting valid larger sequence arrays through an oversized dot. The small distillation witness keeps its previous command layout.

The host rejects nonfinite or nonzero witnesses, malformed/aliased/borrowed results and negative EOS loss. The same absolute-dot overflow bound checks returned audio hidden rows before backward or parameter accumulation. Rejection is terminal and blocks retry/checkpoint/ordinary close while retaining uncertain owners. Forward never accumulates parameters; EOS weight and all parameter/branch scaling are unchanged.

The exact-zero witness enforces an arithmetic diagnostic contract. Numerical parity and generated-signal quality use their existing independent references and tolerances. Native zero-product propagation, precision and resources still need authorised execution on the pinned runtime.

## Tiny complete-step accounting

| Measure | EOSforward | Forwardresults |
| --- | ---: | ---: |
| Tapes | 11 | 11 |
| Boundary tensor downloads | 42 | 42 |
| Downloaded result/gradient elements | 168 | 117 |
| Commands | 3,700 | 3,709 |
| Conservative logical tape bytes | 133,088 | 134,064 |
| Summed planned owned tape bytes | 27,864 | 27,876 |

The forward sequence/full-transformed downloads are replaced by audio hidden and a scalar witness. Accumulation counts are unchanged. Nine commands and 976 summed logical bytes are added; planned owned storage rises by 12 bytes in this fixture. Full snapshots and optimiser diagnostics are unchanged. These are logical/mock counters; native latency, allocator peak and transfer-inclusive throughput are unmeasured.

## Offline verification

Evidence directory: `/workspace/exports/pocket-gpu-training-forwardresults-20261002`.
Gate script: `/workspace/tmp/pocket-forwardresults-gate.sh`.
Final window: **2026-10-02 08:11:43--08:12:50 UTC**.

| Gate | Passed |
| --- | ---: |
| Selected new tests, ten repetitions | 190 top-level + 250 subtests |
| Selected new tests, race detector | 19 top-level + 25 subtests |
| Affected CPU/mock regressions | 447 top-level + 776 subtests |

Final logs contain zero failures/skips. Tape tests compare selected hidden rows and mean with full forward results for 1/2/7 frames and empty/nonempty voice. They check unchanged bindings, exact augmented budgets, dead-invalid complete sources and mismatched topology/extents. Finite-witness tests induce NaN and positive/negative Inf through admitted finite source graphs and accept positive/negative maximum finite values; 16,384/16,385/32,769-element cases exercise bounded chunking.

PTX/Vulkan host mocks cover complete-step objective endpoints, padding, all gradients and parameter/M/V/EMA updates, clipped multistep behaviour, checkpoint/resume/cross-scope rejection and every graph construction failure. Direct calls verify nil private omitted fields, read-only accumulation counts and unchanged public FlowLM outputs. Decoder and evaluator tests reject malformed results, aliases, caller-backed arrays, invalid witnesses and overflow; each evaluator failure prevents retry and retains owners. Independent CPU/F64 and pinned PyTorch mixed-AD/FlowLM/full-step regressions pass unchanged tolerances.

All five functions in `training_forward_results.go` and the shared finite-witness helper have **100% scoped statement coverage**. The selected regression profile covers 57.2% of Pocket TTS and 14.4% of pockettrain; those package figures include unrelated/native paths. Vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain cross-builds, formatting, layout and diff checks pass. No kernel changed.

### Development and review

The first test build had a duplicate failure-test name, corrected before the compiled gates. Inspection found the 16,384-element linear ceiling and the witness was chunked before final validation. No numerical threshold changed.

A mathematical judge timed out after 15 seconds and supplied no approval. Independent source review, native strict-FP/resources/performance and voice/listening acceptance are open. Holds are unchanged.

## Next boundaries

Audio hidden rows still cross the host between transformer and flow tapes. Forward prediction admission, host graph metadata/nonzero constants, dense embedding VJP, diagnostics and full snapshots/live host copies still require work. The EOS mean also remains a host scalar. Continue bounded offline implementation without a full-device or native-speed claim.
