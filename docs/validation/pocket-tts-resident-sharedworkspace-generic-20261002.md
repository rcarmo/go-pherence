## Generic/mock shared resident workspace

`NewSharedWorkspaceResidentState` implements the approved generic/mock-only storage slice: five persistent P/G/M/V/EMA buffers per parameter, one shared scratch/branch/zero/statistics workspace and three four-element globals. No PTX/Vulkan adapter, selectable CLI scope or runtime ceiling/host-guard/download change is included. Existing constructors keep their old allocation and execution paths.

Parent: `a90aec30567d485784f2e04d1803c543c9f2817f`; training source is uncommitted and inference remains staged eight paths / 469 insertions. `@default` conditionally approved this bounded slice in `/workspace/exports/pocket-sharedworkspace-design-review-20261002.md`; source review of the implementation is still required before native adapter or CLI work. All GPU/model/audio/reset/recovery/listening/publication holds remain.

## Optional capability and admission

`ResidentSharedWorkspaceDriver` requires pure complete-allocation inventory validation, bounded guarded prefixes, dense accumulation preflight, exact-bit synchronous copy preflight/execution and pure norm-stat preflight. Global norm and device-scale capabilities are also required. Missing capabilities reject the new constructor before allocation; existing native drivers do not implement the new capability.

`SharedResidentBytes` validates the entire snapshot with the existing numerical inventory checks, then charges `20*elements + 12*largestParameter + 4*largestNormStats + 48`. It preserves shape/state/finite/nonnegative-V and reduction-element ceilings. Logical charge is separate from native/tape/host working-set peak.

The constructor registers each successful allocation immediately and retains a terminal owner on partial allocation, nil-buffer, inventory or initial-upload failure. After allocating all buffers, pure inventory validation checks exact extents, driver/context/allocation physical identity and disjointness before any initial upload. The shared zero receives positive zeros and remains read-only. Snapshot parameter arrays remain caller-owned source data during synchronous construction; no checkpoint format changes.

For two tiny parameters of two and one elements, the new layout charges 144 bytes and owns `5*2+7=17` buffers; measured mock allocation extents match the pure plan. The old global/state-check layout charges 168 bytes. For the released inventory, the same formula gives the previously reviewed 1,839,502,424-byte resident-only charge with 307,981,224-byte headroom below 2 GiB; no released constructor/native execution is performed by these tests. Complete native/host coexistence remains unqualified as documented in the [proposal](pocket-tts-resident-sharedworkspace-proposal-20261002.md).

## Guarded prefix lifetime

The first implementation exposed a raw-buffer resolver. Source review identified that its result could outlive generation or read-only checks. That API was removed.

The final `Prefix` capability receives a private `SharedResidentAccess` token which the driver-created bounded `ResidentBuffer` must capture. Every prefix upload/download and driver operation validates that token before access. The token checks the exact driver pointer, private operation-context pointer, generation, active state, parent-state closed/terminal status and write permission. A new private pointer context forwards caller cancellation for each operation; equal/copied value contexts cannot impersonate it.

No public raw-buffer unwrapping API exists. Prefixes have exact checked extents, cannot close/free parents or become retained tape roots, and may not expose unguarded storage. The fake driver enforces physical range aliases even for distinct wrappers and checks allocation-parent validity. Scratch/branch/stats prefixes are operation-scoped write destinations; zero prefixes are read-only. Retaining a returned prefix interface does not preserve access after the operation ends or another generation starts.

Drivers remain responsible for honouring the capability's token/range/completion contracts, just as existing drivers own native extent/alias/synchronous contracts. No native enforcement has been implemented or qualified. Supported use is caller-serialised; public shared operations set the borrow guard across validation, upload, execution, copy, update/snapshot/close and callbacks to reject reentry.

## Stable gradients and complete preflight

Host accumulation validates every supplied name/extent/value/scale first. It then preflights every dense accumulation and copy-back before the first upload. Bound-tape accumulation gathers all gradient/index roots before preflighting the whole dense/compact pass. Complete source/binding/kernel/combined-live admission still precedes pruning/allocation. Late admission after tape execution retains the consumer and marks state terminal.

Each sorted parameter reads zero when unready or its stable G when ready, writes shared scratch, waits for completed accumulation, then performs completed exact-bit copy-back to persistent G. Only successful copy-back sets readiness/dirty state. Repeated scaled branches and compact embedding contributions use their existing arithmetic; copy adds no scaling or floating-point addition. G identity never swaps with the shared workspace.

Copy failure may partially change G; terminal state prohibits retry/update/snapshot/ordinary close regardless of its ready flag. Earlier gradients are not rolled back. Source tape owners remain live through accumulation/copy-back and are retained on execution/close failures. Missing parameter gradients remain unready until explicitly supplied, including zero contributions.

## Recomputed norm and state checks

Shared-stat preflight covers all P/M/V/EMA post-update checks and both gradient-stat passes, including source/output validity and nonnegative-V mode. Merge/clip and every device-scale update are admitted before optimiser mutation. Pure rejection causes no update or statistics dispatch; numerical state checks still require execution.

Each global pass recomputes a parameter's chunk statistics immediately before its merge. The mode0 maximum survives into mode2; sorted order, previous element counts, A/B ping-pong, fixed maximum and compensated correction are preserved. Shared stats cannot be reused until its merge completes. For N parameters, update performs `2*N` gradient-stat scans plus `4*N` state-check scans. This adds work compared with the legacy cached statistics; no speedup is asserted.

Snapshots remain complete, owned P/M/V/EMA arrays at a completed clean boundary. Restore constructs fresh admitted shared storage. Tests cover multistep clipping and generic snapshot/resume. No native replay/CLI identity is enabled; eventual selectable integration must use a separately reviewed identity and cross-scope policy, preserving legacy hashes.

## Numerical evidence

The host schedule parity test covers 1–4 parameters, 513/257/3/1 element extents, changing maxima, short-after-long reuse, zero/0.25/1 scales, three clipped/unclipped updates and restored snapshots. It compares 33,960 parameter/M/V/EMA values with the established device-scale/state-check mock schedule under the unchanged 2e-6 absolute gate; observed errors are zero. This is a numerical tolerance test, not a byte/hash acceptance gate.

Independent F64 global norms over odd/even parameter counts have relative errors from 1.64e-8 to 4.30e-8, below the unchanged 3e-7 gate. Independent pinned PyTorch fixture `model/pockettts/testdata/training_step_pytorch.json` has schema 1 and revision `0acce6b2f390150267557770d2098c5caa9a18ac`; only its trainable gradient inventory enters resident state.

| Pinned PyTorch P/EMA comparisons | Mean absolute error | RMS error | Maximum absolute error | Absolute gate |
|---:|---:|---:|---:|---:|
| 1,212 | 1.0052768e-9 | 6.26213903e-9 | 1.1920929e-7 | 4e-6, unchanged |

Dense/compact tape tests check stable G, repeated zero/0.25/1 accumulation, primary parameter inventory and explicit compact indices without dense expansion. Copy transport checks preserve negative zero/subnormal bits and untouched tails; exact comparisons are limited to the raw-copy transport contract.

## Failure and permission tests

Named tests are listed in the evidence bundle. Coverage includes unsupported capability/byte rejection before allocation, partial allocation/upload/inventory retention, complete host/tape/compact preflight, mutable-state and second gradient-stat pass preflight, foreign identity/context/expired generation, same-allocation aliases, read-only zero and close rejection, escaped interfaces, callback reentry, incomplete gradient readiness and dirty snapshot rejection.

Faults inject at dense/compact validation, upload, accumulation, copy validation/execution, cancellation during copy completion, stats/merge/clip/update validation/execution, output lookup/download and tape close. Terminal paths reject retries/update/snapshot/close and retain uncertain source/workspace owners. Pure preflight tests confirm no prior upload/update/summary execution at a late rejected branch. Further exhaustive partial-boundary/native failures remain review work; these mock tests do not prove backend completion behaviour.

Initial fixture API/type/expected-value errors were corrected. First source review found raw resolver escape; follow-up found context interface equality could accept copied value contexts. Both were corrected, tested and superseded before final gates. The final bounded review found no blocker in the generic/mock scope, while explicitly retaining driver-enforcement assumptions. It is not native or full-repository approval.

## Final offline gates

Window: **2026-10-02 17:20:39--17:22:52 UTC**. Go network and GPUs disabled, `GOMAXPROCS=2`, `-p=1 -parallel=1`.

| Gate | Top-level passes | Subtests | Failures | Skips |
|---|---:|---:|---:|---:|
| Ten new-slice repetitions | 160 | 320 | 0 | 0 |
| New-slice race run | 16 | 32 | 0 | 0 |
| Affected CPU/mock regression | 1,044 | 1,922 | 0 | 1 |

The existing opt-in `TestTrainingRandomMPFRComparisonDiagnostic` skipped. Vet, host build, ARM64/RISC-V training-test compilation, model-layout and diff checks passed. No kernels, PTX/Vulkan adapters or CLI factories changed in this slice; no native build/run was added. Prior passing resolver/token windows remain archived for provenance, not final acceptance.

Scoped coverage: byte plan 86.7%, access validation 90.9%, operation begin 80%, end 100%, prefix view 63.6%, constructor 88.5%, accumulation 91.7%, stats coordination 87.5%. Untested defensive capability/generation/error paths and actual adapters need further review. Source delta against the sealed handoff milestone is `generic-source.patch`; new files and four shared-conditional generic integration files are included separately from the earlier uncommitted tree.

Evidence: `/workspace/exports/pocket-gpu-training-sharedworkspace-generic-20261002`; gate: `/workspace/tmp/pocket-sharedworkspace-generic-gate.sh`. Submit this bounded slice to `@default` before adapter or CLI work. Current released-size selectable scopes still use the old layout and reject at the 2 GiB ceiling. Full native/host working-set qualification, download removal, voice/listening and publication remain unfinished.
