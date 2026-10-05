## Retained child producers and multi-source edges

`BeginChildTape` retains a read-only child producer under an existing `ResidentTapeLease`. `RunSourcesTape` can then bind pinned roots from multiple exact live owners into one consumer. This supplies the ownership primitive for a prediction tape to survive until backward consumers finish while its transformer producer stays live. No model/CLI scope selects child producers yet; existing host guards and downloads remain.

Source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference remains eight paths / 469 insertions. GPU/model/audio/listening/publication holds are unchanged.

## Owner tree and dependency lifetime

The root lease owns an exact-pointer registry of all live child leases and their complete source-byte charges. Every owner shares the state/root identity, ceiling and allocation policy. A copied or foreign owner cannot act as a registered lease. Public state update/snapshot/accumulation/tape execution/close stay blocked until successful root close.

A child stores its parent and all live edge-source dependencies remaining after pruning. Parents and explicitly borrowed sibling producers cannot close while a dependant exists. A safe premature-close rejection leaves the tree usable. Successful close releases the child producer, removes its registry entry/byte charge, and eventually lets the root close. Root and child native close errors make the whole state terminal. All methods are caller-serialised, including close versus construction/consumer/download/accumulation; driver success must include device completion.

`ResidentLeaseEdge` names an exact live source owner, its original pinned output ID and a constant input ID in the complete consumer. It never propagates a gradient implicitly. `RunTape`/`RunCompactTape` keep their old own-source edge API and delegate to the same multi-owner admission. Current model scopes still use one root and no children.

## Complete admission before allocation

`BeginChildTape` validates the complete graph and dense parameter inventory, pinned declared roots, current resident names/extents, and the combined complete-source ceiling. It limits the tree to 4096 owners, outputs/edges to 4096, rejects duplicate roots, and copies metadata/constants before deferred use. Source leases must have the same exact root/state, be registered/live and expose the pinned root. Edges reject invalid targets, extent mismatch, parameter conflicts, duplicate target IDs and duplicate reshape families within one source.

Backend handoff preflight validates all bound parameters and edge buffers together before pruning/allocation. This also rejects distinct owner IDs that alias the same native storage. Stable pruning remaps parameters, child roots and edge targets. A fully admitted edge pruned away does not create a needless source dependency, although the explicit parent remains pinned.

The new complete source joins every live owner's source charge. Each later synchronous consumer adds its complete source to that total before allocation, even if it has no edges. Borrowed placeholders are conservatively counted. Child close releases its charge; failed owners keep theirs. No native peak-memory or allocator-overhead measurement follows from this logical accounting.

The child enters the root registry before construction so a partial returned tape remains owned on error. Construction, producer run/output, late source admission, consumer execution, accumulation, downloads and cleanup errors poison the state and retain all potentially referenced owners. Cancelled authorised work with a live tree also makes it terminal. There is no rollback or retry of uncertain work.

## Accumulation and numerical checks

A multi-source consumer uses the existing full-source/inventory/pruned runner and whole-pass accumulation preflight. Dense and explicit compact gradients preserve index roots and scale exactly once before one ping-pong swap. Forward root/child construction never accumulates parameter gradients. Endpoint/primary stop-gradient policy remains the caller graph's responsibility.

The lifecycle fixture executes root, prediction child, sibling borrower and a final root-plus-prediction consumer. It makes only one requested consumer download and one explicit accumulation, then closes in order `[4,3,2,1]`. Producer pointers remain live throughout. Nested children, sibling dependencies, re-entry, copied/foreign owners, cross-owner storage aliases, dead edges, mutation/append ownership and partial failures have dedicated tests.

The budget fixture charges root 4B, child 12B and sibling 12B, plus an 8B consumer. A 35B ceiling rejects before allocation; 36B passes. Closing the sibling reduces the live charge from 28B to 16B. Another later-accumulation test verifies earlier gradients remain unchanged after failure, while root, prediction child and failed consumer remain retained and publication/retry/close are blocked.

128 independent F64 comparisons cover two-source values over 64 scales using exact F32 inputs/scales. No producer download occurs; only requested consumer results transfer.

| Comparisons | Mean absolute error | RMS error | Maximum absolute error | Absolute limit |
|---:|---:|---:|---:|---:|
| 128 | 7.79801046e-8 | 9.10922624e-8 | 2.64048577e-7 | 2e-6, unchanged |

The preserved accumulation tests and compact fixture check once-only gradient contributions. These fixtures do not establish released-size model quality, native memory or throughput. Host graph constants/metadata are still copied to freeze retained sources.

## Final offline gates

Window: **2026-10-02 13:51:29--13:53:33 UTC**. GPUs and Go network access disabled, `GOMAXPROCS=2`, tests `-p=1 -parallel=1`. All final commands exited successfully.

| Gate | Top-level passes | Subtest passes | Failures | Skips |
|---|---:|---:|---:|---:|
| Ten repetitions | 880 | 2,120 | 0 | 0 |
| Race detector | 88 | 212 | 0 | 0 |
| CPU/mock regression | 917 | 1,541 | 0 | 1 |
| Compiled-artifact follow-up | 88 | 212 | 0 | 0 |

`TestTrainingRandomMPFRComparisonDiagnostic` is the existing opt-in skip; it did not pass. Vet, host build, Linux ARM64/RISC-V builds, model-layout and diff checks passed. Offline CUDA/PTX and GLSL/SPIR-V compilation/validation passed; this slice adds no new kernel.

Scoped coverage: own-edge mapping, multi-source consumer wrapper and multi-owner binding admission 100%; child constructor 94.9%; existing lease identity/download/retention/close 100%. Child owner-ceiling/allocation-policy and malformed parameter-extent branches remain unqualified. Native execution is unqualified; race gates exercise supported serialised use only.

Evidence: `/workspace/exports/pocket-gpu-training-tapechild-20261002`; script: `/workspace/tmp/pocket-tapechild-gate.sh`. `initial-pass/` preserves the preceding successful broad run. A focused read-only source delegate timed out after 60 seconds with no verdict. Independent source approval was not obtained.

## Prediction integration

A distinct next model scope should retain diagonal/distillation forward child roots, pass explicit condition/input/prediction/desired edges into backward graphs, and close each child after its final consumer/accumulation. Keep host vector downloads and original overflow guards initially; prove exact root/row/provenance mapping, full live-budget admission, primary-only gradients, endpoint input-only stop-gradient, scale-once updates and late failure publication fences before removing transfers. Dense persistent optimizer work, full snapshots, native resources/FTZ/throughput and new-voice listening remain unfinished.
