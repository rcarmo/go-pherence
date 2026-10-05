## Private retained-device diagonal backward

The private diagonal path retains its forward child, checks two scalar admission outputs before backward accumulation, and binds condition/input/desired/prediction as shape-only read-only sources. It downloads no forward vectors. Existing production scopes, redundant host guards and CLI/replay identities are unchanged; the private path has no production caller.

Source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference remains eight paths / 469 insertions. GPU/model/audio/listening/publication holds are unchanged.

## Ownership and admission

`beginDiagonalDeviceSources` builds the existing host-admitted diagonal forward graph, locates its unique condition source through reshape aliases, removes that constant only after complete source/binding validation, and creates a retained child borrowing an already pinned condition root. Root-tree identity, liveness, disjoint edges, parameter inventory, full source and combined live ceiling are admitted by `BeginChildTape`. Ambiguous condition discovery fails closed.

The private provenance record contains the original model identity, exact parent/root/row/time/extents and four source roots. It has a shared single-attempt marker: copied descriptors cannot accumulate twice. All lease operations require caller serialisation; this adds no concurrency support.

The forward child captures value-owned model topology and epsilons and owns copies of time frequencies. Backward uses this same captured graph metadata. Both graphs rebind named parameters from the same pinned resident state; update/restore is prohibited while the lease is live. Tests mutate host frequency/epsilon/topology metadata after forward construction and verify that backward still uses the captured state. Host parameter inventory remains read-only during the private synchronous builders.

`buildDiagonalDeviceAdmission` binds all four exact sources. LSD mode2 checks the retained prediction/desired pair against the existing host-derived cutoff; a finite witness covers condition/input/desired/prediction, including large chunk tails. Two distinct finite scalar zeros are required. No gradients accumulate during this read-only admission. Runtime injection of NaN/Inf or excessive prediction differences rejects before backward construction/accumulation.

`buildDiagonalDeviceTrainingTape` uses a private seed callback in the shared device-source flow builder. Condition/input/desired/prediction have no host arrays. It reads retained prediction and desired for raw loss statistics and the scaled prediction seed, recomputes flow derivative commands from the captured metadata and pinned weights, and preserves `time*dX - scaled prediction seed` target order. Raw statistics are unscaled; each named parameter accumulates once with unit coordinator scale because the callback seed already contains the objective scale. Device source extents remain fully charged before pruning.

The input/desired/prediction roots are minted only from the known private forward builder; shape/count checks precede positional indexing. Semantic ordering is tested against independent CPU composition, not inferred from equal extents. Scalar or result failure clears the returned row and retains the whole tree terminally. Existing gradients may remain after late failure; retry, snapshot, update and close are prohibited. Uncertain owners are retained rather than blindly freed.

Success checks selected result shapes, finiteness, distinct storage and nonnegative square, then closes the exact forward child before publishing the row. The parent remains pinned across rows. Child close failure produces no successful row result. These helpers have no trainer/metrics/replay/checkpoint publication route yet.

## Offline numerical and failure tests

27 combinations cover time 0/0.3/1, scale 0/0.25/1 and logvar -2/0/1. Results and resident parameter gradients match the existing diagonal graph and the independent CPU forward/VJP implementation. F64 loss reference uses the original normalized-square formula with the actual F32 prediction differences.

| Comparisons | Mean absolute error | RMS error | Maximum absolute error |
|---:|---:|---:|---:|
| 10,530 | 8.27725061e-10 | 6.15896965e-9 | 2.38418579e-7 |

Unchanged absolute gates: gradient comparisons 5e-5; loss statistics `2e-6 + 2e-6*abs(reference)`. Generated outputs are compared numerically, without hash acceptance gates.

The tiny private backward graph has 434 commands, 533 tensors and 12,056 full-source logical bytes. Exact budget admits; one byte less rejects. Each successful row makes two mock download calls: two admission elements and eight result elements (condition3/target2/statistics3). No forward vector or parameter-gradient downloads occur before update. Native peak memory, transfer timing and complete-request performance are unmeasured.

Tests cover root/model/row/time/extent/flag provenance, single-use copies, foreign or closed owners, cancellation before and after admission, missing source declarations, malformed or aliased scalar/results, negative square, loss overflow, failed backward construction, metadata drift and child-close failure. Post-admission result/close errors may follow accumulated gradients; terminal publication fencing is verified. Large finite witness tests detect an invalid last chunk and accept maximum finite values without reduction overflow.

Initial fixture errors (graph type/zero materialisation, loss return type, mock call counts and close-target selection) were corrected before final passing gates. Intermediate logs are retained.

## Final offline gates

Window: **2026-10-02 15:27:24--15:29:13 UTC**. GPUs and Go network access disabled; `GOMAXPROCS=2`, tests `-p=1 -parallel=1`.

| Gate | Top-level passes | Subtest passes | Failures | Skips |
|---|---:|---:|---:|---:|
| Ten repetitions | 470 | 1,700 | 0 | 0 |
| Race detector | 47 | 170 | 0 | 0 |
| CPU/mock regression | 985 | 1,704 | 0 | 1 |
| Compiled-artifact follow-up | 47 | 170 | 0 | 0 |

The existing opt-in `TestTrainingRandomMPFRComparisonDiagnostic` skipped. Vet, host build, Linux ARM64/RISC-V builds, model layout, diff checks and offline PTX/SPIR-V compilation/validation passed. Kernels and ABI are unchanged. Race tests exercise supported serialised ownership.

Scoped coverage: edge mapping, metadata/source-ID admission, scalar graph and metadata freeze 100%; backward builder 88.6%, runner 90.7%, owner constructor 88.6%, forward-root validation 92.3%. Some defensive internal-builder errors and graph-discovery failure branches are not directly injected; native execution remains unqualified.

Source review first raised unsupported concurrency, prediction/recomputation consistency, retained failures and reshape-only discovery. Frozen graph metadata and a drift test strengthened same-state consistency. Serialisation, deliberate retention and fail-closed discovery were retained. Follow-up review found no further issue on those contracts and requested nil begin-context and positional boundary checks; both were added and tested. A first follow-up attachment attempt failed path inspection and returned no verdict. Final hardened code has not received another independent source verdict or native acceptance.

Evidence: `/workspace/exports/pocket-gpu-training-diagonaldeviceinputs-20261002`; script: `/workspace/tmp/pocket-diagonaldeviceinputs-gate.sh`. Prior successful pre-hardening windows are under `initial-pass/` and `pre-boundary-pass/`.

## Next work

Distillation needs a corresponding retained-owner/primary-endpoint/seed contract with primary-only parameter gradients, scale-once and endpoint stop-gradient. Production opt-in integration must preserve current host guards until source/native qualification permits a new download policy. Hidden/forward/backward downloads in existing scopes, coherent snapshots and dense persistent optimiser work remain. No native training, audio or listening acceptance has run.
