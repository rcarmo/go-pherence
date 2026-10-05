## Compact indexed embedding VJP primitive

Embedding mode three produces compact active-token gradients instead of materialising a vocabulary-sized dense table. It uses one sequence-ordered reduction per active token and explicit zero padding. This is a primitive-only slice: current resident bindings still require parameter-sized gradients, and no Pocket TTS builder, CLI choice or replay scope uses mode three.

The source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference remains eight paths and 469 insertions. All execution used offline CPU/mock qualification with NVIDIA/Vulkan disabled. CUDA/PTX and GLSL/SPIR-V compilation and inspection were offline only. No native GPU, released-model/audio, recovery or publication work occurred.

## Index and reduction contract

`BuildCompactEmbeddingIndex` accepts one to 1,024 finite, exact F32 token IDs in a vocabulary up to 65,536. It sorts a copied token array and deduplicates it, without changing the original IDs. Active tokens occupy ascending slots; each bucket's occurrence rows follow their original sequence order. Padding slots contain token -1 and empty buckets.

The packed F32 index has `3*rows+1` elements:

1. `rows` token slots, with trailing -1 padding;
2. `rows+1` bucket offsets, from zero through `rows`;
3. `rows` original occurrence row IDs.

Whole-index admission checks dimensions, finite integer bounds, endpoint offsets, monotonic extents, strictly increasing active tokens, nonempty active buckets, trailing-only inactive slots, empty padding buckets and globally unique, sequence-ordered row indices. Offsets traverse exactly `rows` entries; uniqueness and row bounds prove complete coverage. Every admitted integer is exactly representable in F32. The full index must be an explicit constant in a tape, and the original full-source logical ceilings still apply before pruning.

Each kernel output element owns one slot/column reduction. Active slots sum only their original occurrence rows in sequence order; inactive slots write positive zero. No atomics or reassociation is introduced. The output has `rows*width` elements, independent of vocabulary. Scattering active slots to a dense table recovers the previous VJP; absence/padding produces zero. Dense parameter accumulation is not part of this primitive.

Modes zero, one and two preserve their old layouts/extents/order and current CLI routes. Mode three reuses the existing embedding entrypoint ABI but admits a different second-input extent and compact output. Raw device callers must admit immutable index contents; wrappers check shapes, owners and aliases, while kernels guard offset/row/token bounds before conversion/indexing. Malformed raw indices are not supported numerical inputs.

## Storage and setup costs

With vocabulary32,768, width1,024 and rows512:

| Storage | Indexed dense mode2 | Compact mode3 |
| --- | ---: | ---: |
| VJP output bytes | 134,217,728 | 2,097,152 |
| Packed index bytes | 133,124 | 6,148 |

That is a theoretical **64-fold smaller VJP output**, not a native allocation or peak-RSS measurement. Source row gradients, table parameters, resident gradient buffers, global norm/optimiser state and later accumulation still exist. When rows exceed vocabulary, padded compact output can be larger than dense output; integration must select its contract deliberately rather than assume every shape saves memory.

The compact builder uses a rows-sized integer workspace and stable repeated scans of IDs after sorting. This avoids vocabulary-sized host offsets, but trades work for memory: at most rows distinct token scans, bounded at1,024 rows. It does not cache indices. Five 100-iteration CPU benchmark samples at vocabulary32,768/rows512 measured **10,624 B/op and two allocations**, with **108.6--120.5 microseconds/op**. The dense index builder in the same run measured about **409,600 B/op and two allocations**, with **102.1--109.3 microseconds/op**. Compact construction is much smaller but not faster in that fixture; this is index setup only, not full-request or native performance.

The primitive writes at most rows*width compact slots. It still needs a later accumulation interface that accepts active-token slot gradients; the current resident binding validator rejects a gradient extent different from its parameter. No binding rule was relaxed in this slice.

## Numerical and structural evidence

Independent row-scatter F64 references are compared against a compact slot-gather formula for small, repeated-token, absent-token and maximum1,024-row cases. Across **3,102 output comparisons**, absolute F32/F64 error has mean **2.92727115e-12**, RMS **3.57174686e-11** and maximum **6.98491931e-10**. The existing `2e-6*max(1,abs(reference))` gate is unchanged. This fixed corpus does not establish broad error characterisation or native rounding/FTZ acceptance.

A cancellation-sensitive bucket separately checks sequential-F32 reduction order, which is the arithmetic contract. Pocket TTS mock interpretation scatters compact outputs and compares with legacy dense VJP at existing tolerance, checks zero padding and verifies pruning preserves the compact output. No generated-output hash or byte-identity gate is used. Compiled kernels are not numerically executed.

Admission tests cover bad IDs/shape/vocabulary/row bounds, nonfinite/fractional/decreasing offsets, endpoints, token ordering/padding, empty active buckets, invalid/duplicate/unordered rows, nonconstant index tensors, wrong output shape/mode, dead-source NaN and complete index byte budgets. Independent/concurrent construction owns its index and preserves the source.

PTX mock tests exercise mode-three ABI/grid/extents and tape dispatch, short-index rejection, terminal launch/sync poisoning and retry prevention. Vulkan mock tests cover stage/tape push/groups/extents, short-index rejection and cancellation. Existing owner/alias, preflight, terminal-retention and mode-zero/one/two tests remain in the regression gate. The shader remains inside the conservative SPIR-V opcode envelope; no reflection admission was widened.

## Offline gates

Evidence directory: `/workspace/exports/pocket-gpu-training-embeddingcompact-20261002`.
Gate script: `/workspace/tmp/pocket-embeddingcompact-gate.sh`.
Window: **2026-10-02 10:22:39--10:24:19 UTC**.

| Gate | Passed | Skipped |
| --- | ---: | ---: |
| Selected tests, ten repetitions | 180 top-level + 50 subtests | 0 |
| Selected tests, race detector | 18 top-level + 5 subtests | 0 |
| Affected CPU/mock regressions | 733 top-level + 1,216 subtests | 1 |
| Compiled-artifact follow-up | 18 top-level + 5 subtests | 0 |
| Numerical follow-up | 10 repetitions | 0 |

Zero failures. The existing opt-in `TestTrainingRandomMPFRComparisonDiagnostic` remains skipped with `requires separately reviewed fixed-corpus error-characterisation gate`. It is unrun, not a passed test; no restriction changed.

Compact index builder, validator and both embedding runtime wrappers have **100% scoped statement coverage**. Package coverage under this selector is88.2% training,12.5% NVIDIA runtime,35.1% Vulkan,68.7% Pocket TTS and14.7% pockettrain. Vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain cross-builds, formatting, model-layout and diff checks pass.

Offline CUDA12.8 sm80/PTX8.7 compilation uses `--fmad=false` without fast math; `ptxas` accepts the module. GLSL compiles, passes `spirv-val` and conservative reflection (existing bindings/push/local size). These checks are not hardware compatibility, numerical parity or speed acceptance.

## Review and next work

The invariant judge timed out after15seconds. No independent source or native approval is claimed. Next implement a separately admitted compact-gradient binding/accumulation contract, preserving complete inventory admission, parameter scaling exactly once, sequence-order reduction, owner retention and output lifetimes. Dense optimiser buffers/work remain unless separately addressed.

Transformer/flow handoffs, prediction admission, EOS diagnostics, full snapshots and authorised native resources/performance/voice acceptance are still open. All execution/recovery/listening/publication holds remain. This primitive does not complete the overall implementation goal.
