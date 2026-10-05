## Compact resident tape bindings and coordinator

The resident coordinator now accepts compact embedding gradients through a separate explicit API. It retains the compact gradient and its immutable index through pruning, preflights the complete mixed dense/compact accumulation pass and calls the optional compact driver without dense tape expansion. No Pocket TTS builder, CLI choice or replay scope selects this API yet.

The source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference remains eight paths and 469 insertions. Every execution used offline CPU/mock qualification with NVIDIA/Vulkan disabled. Shader/PTX compilation was offline only. No native GPU, released-model/audio, recovery or publication work occurred.

## Separate binding admission

`ResidentCompactEmbeddingBinding` supplies index ID, rows, width and vocabulary alongside an ordinary parameter/gradient ID pair. `ValidateResidentCompactTapeBindings` first validates the complete tape and the common binding inventory, then admits the compact descriptors. Compact entrypoints require a nonempty descriptor map; nil does not silently select dense semantics.

Common checks retain valid distinct parameter/gradient IDs, unique bound parameter IDs, declared gradient roots, constant parameter sources, names and inventory ceilings. Dense bindings retain parameter-sized gradient extents. Compact entries require parameter `vocabulary*width`, gradient `rows*width` and index `3*rows+1`, under existing embedding limits. The index must be an explicit constant and a declared output, distinct from gradient and every rebound parameter. Its complete token/offset/occurrence content passes the existing compact index validator.

A compact gradient must be produced directly by a matching mode-three embedding command using that exact index and dimensions. Constant gradients, unrelated producers and indirect reshape producers are rejected by the explicit compact contract. Ordinary dense binding admission rejects mode-three gradients and reshape aliases even when rows equals vocabulary and their element counts happen to match. Layout is never inferred from extent alone.

A lazy marker map tracks mode-three outputs/reshape aliases; graphs without mode three do not allocate that map. The shared validator gains a command scan, so admission overhead is not claimed to improve. It does not recursively infer compact layouts through arbitrary arithmetic; the supported compact contract requires a direct producer.

All source validation, descriptor checks, resident parameter inventory/extents, requested-output declaration and driver capabilities precede tape allocation. Invalid dead tensors and full source logical ceilings remain enforced before pruning. Unknown resident names and late resident extent mismatches produce no driver mutation.

## Roots, accumulation and ownership

`RunCompactZeroPrunedTape` explicitly requires zero-constant/lifetime/tape support and `ResidentCompactEmbeddingDriver`. It uses the existing resident execution lifecycle rather than an independent owner implementation. The whole source is admitted before pruning.

Pruning roots comprise requested downloads and, when accumulating, every binding gradient plus every compact index. Stable tensor mappings update gradient and index descriptors together. Parameters unused by the VJP can be removed from the tape (mapping -1); their resident entries remain identified by the already-admitted name. Declared index roots pin immutable storage so borrowed index/output buffers remain available through accumulation.

After synchronous tape execution, a first loop fetches all dense/compact gradient buffers and compact index buffers, selects the prior gradient or initial zero, and validates every accumulation. No gradient ping-pong swap or ready/dirty mutation occurs until that complete loop succeeds. A second loop dispatches the appropriate dense or compact driver and swaps each parameter's gradient/scratch exactly once. Scale is applied once after the sequence-ordered compact VJP sum. Absent tokens retain the accumulator's `previous+scale*0` formula.

Source tensors/commands/output arrays and caller descriptor maps are not mutated. Remapped descriptor maps are coordinator-owned. The borrowed flag blocks snapshot, update, host accumulation and Close during execution. Tape construction/run/output/download/accumulation/cleanup failures retain uncertain owners and become terminal with no retry. Pure post-run preflight rejection closes the unmodified tape and leaves gradients unchanged, matching existing dense behavior; it does not fabricate a native execution failure.

Nonaccumulating calls do not retain otherwise unused compact roots, perform no parameter accumulation and leave the state clean. A dirty successful accumulating state rejects snapshot until a complete update. Existing ordinary dense APIs retain their signatures and pass nil compact metadata internally.

## CPU/mock evidence

A purpose-built CPU driver supports the minimal embedding/vector fixtures without opening hardware. Tests cover a mixed dense/compact inventory, repeated accumulation at scales zero/.25/1/-.5, independent source snapshots, unchanged downloaded boundaries, gradient+index roots, a pruned unused parameter, exactly-once swaps/calls and clean completed update/snapshot. Dense persistent buffers are still present.

Admission tests cover nil/empty metadata, unknown names, index IDs/declaration, rebound indices, dimensions/extents, mismatched direct provenance, missing producers, invalid dead source, byte budget, finite scale, undeclared requested outputs, resident inventory and missing compact-driver capability. Same-size compact and reshape cases cannot masquerade as dense bindings.

Failure injection exercises compact and dense preflight, compact and dense accumulation execution, index/gradient output fetch, construction/nil tape, run, download/count/extent/nonfinite output and cleanup failures. Both preflight failures prove zero accumulation calls and no dirty/terminal state; execution failures prove retained ownership, rejected Close and no retry. Borrow hooks attempt snapshot, host accumulation, update and Close during execution and confirm all are blocked.

These tests validate coordinator routing/arithmetic and lifetime metadata, not native shader numerical execution or allocator peaks. No complete-request speedup, released-size memory savings or voice quality claim is made. The mathematical storage opportunity from compact VJP remains separate from persistent dense gradient/optimiser buffers and their dense work.

## Offline gates

Evidence directory: `/workspace/exports/pocket-gpu-training-compactbindings-20261002`.
Gate script: `/workspace/tmp/pocket-compactbindings-gate.sh`.
Final window: **2026-10-02 10:49:11--10:50:04 UTC**.

| Gate | Passed | Skipped |
| --- | ---: | ---: |
| Selected tests, ten repetitions | 350 top-level + 410 subtests | 0 |
| Selected tests, race detector | 35 top-level + 41 subtests | 0 |
| Affected CPU/mock regressions | 746 top-level + 1,241 subtests | 1 |
| Compiled-artifact follow-up | 35 top-level + 41 subtests | 0 |

Zero failures in final gates. The existing opt-in `TestTrainingRandomMPFRComparisonDiagnostic` remains skipped with `requires separately reviewed fixed-corpus error-characterisation gate`. It is unrun, not a passed test; no restriction changed.

Public compact validator/runner have100% scoped statement coverage; internal compact validation93.5% and shared execution98.8%. Some defensive index/provenance/pruning error paths are not independently reachable under the preceding complete-tape admission. Coverage is not independent approval. Existing pinned CPU/F64/PyTorch model gradients/updates, replay, clipping and terminal-retention tests pass unchanged tolerances. No generated-output hashes or byte identity are used as signal acceptance gates; source/replay structural checks remain exact where required.

Vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain cross-builds, formatting, layout and diff checks pass. Offline CUDA12.8 sm80/PTX8.7 `--fmad=false`/ptxas, GLSL/spirv-val and conservative shader reflection also pass; this slice changes no kernels. Initial test-driver field/fixture mistakes were corrected and failed development logs retained.

## Review limits and remaining work

An invariant judge found no concrete blocker in the stated contract, but emphasized arithmetic bounds, index/output aliases, stable pruning/provenance, owner lifetime, whole-pass preflight and exactly-once ping-pong swaps. Existing bounds and the targeted tests above cover those conditions. This judge did not inspect source or execute native code; no independent source/native approval is claimed.

Next implement a separately qualified Pocket TTS compact rewrite and CLI/replay scope with public/legacy builders preserved, complete source admission before rewrite and combined compact admission before dispatch. Dense gradient buffers and optimiser work, transformer/flow handoffs, prediction admission, EOS diagnostics and snapshots still remain. All native execution/recovery/listening/publication holds remain unchanged. This coordinator slice does not complete the overall implementation goal.
