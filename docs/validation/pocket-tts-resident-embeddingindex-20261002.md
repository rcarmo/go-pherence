## Pocket TTS indexed embedding VJP primitive

A new embedding mode provides sequence-ordered indexed reductions without scanning every token row for every dense output element. It is a primitive-only slice: existing Pocket TTS builders, CLI choices and replay scopes still use modes zero/one. No full-device or sparse-gradient claim follows from this implementation.

The source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference remains eight paths and 469 insertions. NVIDIA/Vulkan runtime execution was disabled for every gate. CUDA/PTX and GLSL/SPIR-V were compiled and inspected offline, not loaded on hardware. No released-model/audio, recovery or publication work occurred.

## Representation and admission

`BuildEmbeddingIndex` accepts one to 1,024 F32 token IDs and a vocabulary of one to 65,536. IDs must be finite, exact integers in range. It creates vocabulary+1 prefix offsets followed by exactly one row index per token occurrence. Rows within each token bucket remain in original sequence order. No source array is borrowed or mutated; independent/concurrent calls own their indices.

`ValidateEmbeddingIndex` admits the complete explicit constant before a tape can dispatch. It checks the full shape, zero/terminal offsets, finite integer bounds, monotonic offsets, strictly increasing rows within each bucket and global row uniqueness. Monotonic offsets from zero through `rows` traverse exactly `rows` entries; uniqueness and row bounds therefore establish complete row coverage. No unused/dead malformed tensor is hidden by pruning.

All index integers are exactly representable in F32 under existing bounds. The `embedding` operation's mode two retains two inputs and the same kernel ABI, but its second input has `vocabulary+1+rows` elements rather than `rows`. Modes zero/one retain their original extents, operation order and raw-ID interpretation. Complete-source tensor/command/logical-byte ceilings still apply to the larger constant index.

Raw PTX/Vulkan entrypoints validate shapes/owners/aliases, not device buffer contents; raw callers must supply an admitted immutable index, just as legacy raw callers must supply valid IDs. Tape callers receive complete constant-content admission. Kernels additionally guard offset and row ranges/integrality before indexing/conversion, but raw malformed indices are not a supported numerical input contract.

## Kernel behaviour and scope

Each output element still owns one dense table-gradient reduction. Mode two reads that token's offset pair, walks only its row occurrences and sums in sequence order. Empty buckets write positive zero. No atomics, cross-thread reduction or reassociation were introduced. The source summands and their addition order match mode one, including cancellation-sensitive repeated tokens.

This removes unrelated-row comparisons from the VJP algorithm; it does not remove dense output allocation/initialisation, dense resident accumulation or optimiser work. Dense output remains `vocabulary*width` F32 values. Native speed and hardware rounding/FTZ behaviour remain unqualified.

For illustration, vocabulary 32,768, width 1,024 and 512 rows imply 17,179,869,184 legacy token-row comparisons across dense elements. The indexed algorithm reads 67,108,864 offset values and 524,288 row occurrences/source summands, while still writing all 33,554,432 dense outputs. These are algorithmic work counts, not timing or bandwidth measurements. GPU caches, scheduling and transfer/setup costs can change the outcome.

The packed index adds `4*(vocabulary+1)` logical bytes relative to legacy IDs: 131,076 bytes in that example. Its total is 133,124 bytes. Host construction needs an integer prefix workspace plus the F32 index. A five-sample 100-iteration CPU benchmark at vocabulary 32,768 and 512 rows measured two allocations and about **409,600 B/op**, with **105--110 microseconds/op** (median 107.7 microseconds). This is index construction only, not full training or native VJP performance. Integration must account for index upload/retention, full admission and dense output cost.

## Numerical and structural evidence

Independent test code scatters source rows into an F64 dense reference, while the indexed formula gathers only listed occurrences. Small, repeated, absent-token and maximum-row fixed cases pass the existing `2e-6*max(1,abs(reference))` comparison. Across 120 output comparisons, measured absolute F32/F64 error has mean **1.28056854e-10**, RMS **3.19524185e-10** and maximum **1.86264515e-9** for that bounded corpus. This is not a broad error characterisation or grounds to change tolerances.

A cancellation-sensitive repeated-token fixture separately checks the actual sequential-F32 addition contract; it does not use a hash or generated-output byte identity. Pocket TTS test interpretation compares legacy and indexed tapes numerically and checks pruning preserves the output. Those CPU formula tests do not execute compiled kernels or prove native parity.

Tests reject negative/fractional/nonfinite/out-of-range IDs, wrong vocabulary/row bounds, index lengths, bad endpoints, decreasing/fractional/nonfinite offsets, invalid/duplicate/out-of-order rows, nonconstant tape indices, wrong shapes/modes, late dead-source NaN and complete-index byte ceilings. Ownership and concurrent index construction tests pass.

PTX mock tests cover the unchanged seven-argument ABI and grid, larger index extent, tape dispatch, owner/alias/short-index rejection, kernel preflight, terminal launch/sync poisoning and retry rejection. Vulkan mock tests cover stage/tape dispatch, groups/push words, larger index extent, owner/alias/short-index rejection and cancellation. Existing terminal owner-retention and legacy regressions pass.

## Offline gates

Evidence directory: `/workspace/exports/pocket-gpu-training-embeddingindex-20261002`.
Gate script: `/workspace/tmp/pocket-embeddingindex-gate.sh`.
Final full gate: **2026-10-02 09:53:21--09:54:56 UTC**.

| Gate | Passed | Skipped |
| --- | ---: | ---: |
| Selected tests, ten repetitions | 90 top-level | 0 |
| Selected tests, race detector | 9 top-level | 0 |
| Affected CPU/mock regressions | 708 top-level + 1,180 subtests | 1 |
| Compiled-artifact follow-up | 9 top-level | 0 |
| Extended fixed numerical follow-up | 10 repetitions + 1 race test | 0 |

Zero failures in final gates. The existing opt-in `TestTrainingRandomMPFRComparisonDiagnostic` remains skipped with `requires separately reviewed fixed-corpus error-characterisation gate`. It is unrun, not a passed test; no restriction changed.

Embedding shape, index builder, index validator and both runtime embedding wrappers have **100% scoped statement coverage**. Package coverage under the regression selector is 87.5% for training, 12.4% for NVIDIA runtime, 35.1% for Vulkan, 68.6% for Pocket TTS and 14.4% for pockettrain. Vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain cross-builds, formatting, model-layout and diff checks pass.

Offline CUDA12.8 compilation targets sm80/PTX8.7 with `--fmad=false`, no fast math. `ptxas` accepts the module. GLSL compiles with the existing descriptor/push/local-size contract, passes `spirv-val` and conservative `InspectVulkanShader` reflection. Reflection is not a numerical or hardware acceptance gate.

The first compiled GLSL version used logical-and/unordered-not-equal instructions outside the repository's closed opcode envelope. Reflection caught it. Guards were rewritten as staged comparisons with conversions after bounds, without widening shader admission. Tests were rerun against the final compiled artifact. Earlier fixture errors (duplicate-row test initially unchanged, incorrect interpreter output-position indexing and pruning signature) were corrected; development failures are archived and are not current blockers.

## Review and next work

The focused source judge timed out after 60 seconds; the bounded invariant judge also timed out. Neither is approval. Independent source/native review, resources/performance and new-voice/listening acceptance remain open.

Next integrate the indexed primitive only through a separately qualified resident scope, preserving public/legacy outputs, complete source validation, parameter scaling and exactly-once accumulation. Dense table allocation and accumulation remain a larger completeness issue. Transformer/flow handoffs, prediction admission, EOS diagnostics and full snapshots are still open. This offline primitive releases no hold and does not complete the implementation goal.
