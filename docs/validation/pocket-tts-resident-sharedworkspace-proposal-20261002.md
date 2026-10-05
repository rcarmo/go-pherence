## Shared resident workspace proposal

A five-buffer-per-parameter layout plus one shared workspace would reduce the released configuration's calculated resident charge from **2,866,551,644** to **1,839,502,424 bytes**, leaving **307,981,224 bytes** below the existing 2 GiB CLI maximum. This proposal is unimplemented and does not establish safe native peak memory or released-size training. Runtime limits, current constructors, host guards and downloads remain unchanged.

`@default` authorised a bounded source-level design proposal only after the [released-resource audit](pocket-tts-resident-releasedresources-20261002.md). No runtime implementation, ceiling widening, model/native/GPU/reset/recovery/audio/listening/publication or download-policy change is authorised. Implementation ownership stays with `@go-pherence`; design approval is still required.

Parent: `a90aec30567d485784f2e04d1803c543c9f2817f`. Reviewed coordinator/private helpers have only the requested comment corrections; the proposal adds isolated test arithmetic and this design. The inference index remains eight paths / 469 insertions.

## Allocation inventory and headroom

Let E be total trainable elements, N the largest parameter, and Smax the largest `NormStatsShape`. The independently checked released inventory has E=89,448,706, N=4,194,304 and Smax=49,152. Several transformer FFN matrices share the maximum; there is no unique largest parameter.

| Allocation | Current | Proposed |
|---|---|---|
| Parameter P | Per parameter | Per parameter |
| Gradient G | Per parameter | Per parameter; stable destination |
| Accumulation scratch | Per parameter | One N-element buffer |
| Host branch upload | Per parameter | One N-element buffer |
| Immutable zero | Per parameter | One N-element buffer |
| Adam first moment M | Per parameter | Per parameter |
| Adam second moment V | Per parameter | Per parameter |
| EMA | Per parameter | Per parameter |
| Norm statistics | Per parameter | One Smax-element buffer |
| Global A/B/clip | Three four-element buffers | Unchanged |

All allocations are physically separate. No optimiser state, parameter or gradient destination aliases another. The zero buffer remains positive zero for its entire life; it is neither scratch nor an upload destination. Host branch uploads touch only the active prefix of their separate buffer.

| Proposed released inventory | Bytes |
|---|---:|
| Five persistent buffers: P/G/M/V/EMA, `20*E` | 1,788,974,120 |
| Shared scratch/branch/zero: `12*N` | 50,331,648 |
| Shared norm statistics: `4*Smax` | 196,608 |
| Global buffers | 48 |
| Total logical resident | 1,839,502,424 / 1.713170 GiB |
| Reduction from current | 1,027,049,220 |
| Headroom below 2 GiB | 307,981,224 / 293.714 MiB |

Removing only branch/zero without shared scratch/statistics leaves a six-buffer floor of 2,150,961,996 bytes and still fails the current ceiling. This proposal therefore requires the copy-back and recomputation contracts below; deleting two arrays is insufficient.

### Coexistence with tapes

Keep the current independent resident ceiling and conservative all-live-complete-source tape ceiling. Do not count a borrowed resident parameter twice in a physical allocation estimate, and do not subtract its extent from source admission. Logical admission and native peak are separate quantities.

The verified distillation forward/backward complete-source lower bound is 476,038,888 bytes. Adding it conservatively to the proposal gives 2,315,541,312 bytes, 168,057,664 above a hypothetical *combined* 2 GiB envelope, before the transformer parent or row-condition child. That hypothetical envelope is stricter than today's separate resident/tape ceilings. A total below one resident cap alone is insufficient for a memory-policy decision.

Synthetic released-dimension pruning/zero/lifetime plans, subtracting only unique borrowed parameter/external slots, give 55,756 forward-owned bytes and 39,313,972 backward-owned bytes. Their sum plus proposed resident leaves 268,611,496 bytes below 2 GiB. This calculation excludes transformer/EOS parent, row-condition children, other live consumers, allocator alignment, modules/pipelines/plans and command staging. It uses synthetic zero fixtures, not native buffers or model values; it cannot establish complete-request physical peak.

For sensitivity only, reserving 64/128/256/512 MiB beyond resident leaves 240,872,360 / 173,763,496 / 39,545,768 / -228,889,688 bytes respectively. These scenarios are not measurements. Before native qualification, record exact parent/condition/consumer coexistence for admitted released frame/prefix geometries, then measure allocation/retention/driver overhead separately when authorised.

### Coexistence with host state

One complete F32 P/M/V/EMA set is 1,431,179,296 bytes. The live trainer, backend completed-state boundary and freshly downloaded snapshot can coexist at successful synchronisation: at least 4,293,537,888 bytes across those three sets. Graph metadata, input/sample arrays, host gradients, nontrainable buffers, conversion dictionaries, serialization, alignment and allocator overhead are additional. Preparation/restore has different copy lifetimes and must be audited independently. The workspace proposal changes none of those host ownership contracts.

Checkpoint and snapshot are only permitted after every lease/consumer closes and a complete update succeeds. Thus device tapes cannot coexist with snapshot publication by contract, but host metadata/scratch retention still needs separate accounting. Do not report host GC or logical counters as peak RSS.

## Proposed capability and prefix-view rules

Keep existing constructors and legacy buffer layouts unchanged. Add an explicit optional shared-workspace driver capability and a separate constructor/scope only after approval. The capability must expose:

* a bounded prefix view of an owned buffer, with exact logical element count;
* pure extent/liveness/disjoint-range copy preflight;
* synchronous device-to-device exact-bit copy, with failure/completion evidence.

These are proposed contracts, not existing APIs. NVIDIA has a raw copy helper, but it needs resident-driver context/owner/extent validation and synchronous error/completion handling. Vulkan `ReshapeF32` requires the same element count and cannot express a smaller prefix today; an explicit bounded view and device copy operation are prerequisites. Do not emulate copy with a scaled vector add: adding zero can change signed zero and contaminate numeric equivalence. A raw copy preserves the computed F32 representation; arithmetic outputs still use independent-reference tolerances.

For each parameter n, admit `0<n<=N`, checked `4*n`, and `stats(n)<=Smax`. A prefix starts at offset zero, has exactly n elements, and cannot enlarge capacity or expose stale tail elements. Buffers may not be freed, uploaded, downloaded, updated or sliced through an expired parent. Aliasing checks use physical byte ranges and owner identity, not merely wrapper pointer equality. Different views of the same storage overlap and must reject in accumulation/update/copy preflight.

Views own metadata only and never free parent storage. They remain valid through synchronous completion, are not exposed as tape output roots, and expire with the resident owner. The persistent gradient buffer identity stays stable. A tape may retain its own output source through accumulation/copy-back; shared destinations cannot enter any retained producer graph. The state remains caller-serialised; active borrow/lease guards reject public mutation, snapshots, close and reentry.

## Admission, accumulation and copy-back

Before allocation, validate the complete snapshot/parameter inventory, finite state, nonnegative V, limits, required driver capabilities, all persistent/shared extents and arithmetic ceilings. Register every successful allocation immediately. Any later allocation/upload failure returns a retained terminal owner, as today; no blind cleanup of uncertain resources.

Before a host accumulation pass, validate every branch name, extent, finite value and scale, every prefix view and copy-back destination. Preflight all branches before the first upload. Before a bound/compact tape, keep complete source/inventory/kernel/all-live admission before pruning/allocation; after execution, validate every gradient source, compact index and all accumulation/copy-back calls before the first persistent gradient mutation.

For each sorted parameter:

1. Select immutable zero prefix when `ready=false`, otherwise persistent G.
2. Upload host gradient to shared branch prefix, or borrow the completed tape's dense/compact gradient source.
3. Compute `scratch = previous + scale*source` using the existing numerical/compact contract, without aliasing any input.
4. Await device-complete accumulation before copying scratch to persistent G.
5. Await device-complete copy-back; only then set this entry `ready=true` and the state `dirty=true`.
6. Reuse workspace only after both operations complete. Keep tape/producer owners live until the entire pass and close complete.

Copy-back replaces today's per-entry G/scratch pointer swap. Repeated branches still read the committed persistent G, and scale applies exactly once; copy introduces no new arithmetic. Zero-scale and previously ready entries must retain the existing signed-zero/multiply/add policy. Compact embedding accumulation still uses sorted active token slots, sequence-order sums, and the original scale; no sparse optimiser semantics are added.

### Failure points

| Point | Required state |
|---|---|
| Pure complete admission rejection before allocations/borrow | No mutation or owner leak; retry only where the existing API allows it |
| Partial construction / source upload failure | Register/retain all created buffers, terminal state |
| Driver preflight failure after a pinned producer executed | Whole lease tree terminal; retain producer and any consumer |
| Accumulation failure | Scratch and all source owners retained; persistent G may be unchanged, but no retry |
| Copy-back failure or uncertain completion | Persistent G may be partially changed; retain workspace/source/owners, no rollback or retry |
| Cancellation between accumulation/copy/next entry | Terminal after work starts; workspace not reusable |
| Later branch failure | Earlier gradients remain; no update/snapshot/checkpoint/result publication |
| Consumer or parent close failure | Terminal owner/tree retained, no release of dependencies |
| Optimiser/post-check/snapshot/conversion failure | Preserve existing terminal retention/publication rules; no successful step increment or checkpoint |

A copy failure cannot be hidden by an unchanged `ready` flag. Terminal state dominates all readiness/dirty flags. No generic recovery/reset/free-on-error path is part of this proposal.

## Shared norm statistics and numerical order

The existing global reduction computes all per-parameter statistics once, then merges mode0 maxima and mode2 fixed-max scaled sums. With one shared statistics buffer, each mode recomputes an entry's statistics immediately before its merge. This doubles per-entry norm-stat scans for clipping; reduced memory comes with extra dispatch/work, not an assumed speedup.

Preflight the complete sorted max pass, fixed-max pass, clip operation and every optimiser update before any optimiser state mutation. Preflight all prefix/stat extents before the first summary upload/dispatch; returned summaries remain four scalars. Shared stats must not be overwritten until its consumer merge completes. Gradients remain immutable throughout both passes; no copy-back/accumulation may interleave.

The maximum from mode0 remains fixed for all mode2 entries. Preserve sorted parameter order, chunk size 256, intra-chunk evaluation and compensated merge order. Same inputs/kernels should produce the same rounded summary, but acceptance uses independent F64 distributions and unchanged calibrated tolerances; no generated-summary hash or byte-equality gate. Test increasing maxima, many chunks, cancellation and stale-tail sentinels. FTZ/DAZ/nonfinite behaviour remains a separate native gate.

Legacy host-F64 norm states need only each entry's statistics/download before reusing shared storage. Device-state checks already consume P/M/V/EMA one source at a time: use the same bounded stats prefix, complete each merge, and require nonnegative V. Post-update checks must cover every persistent state array before step publication.

## Ready, dirty, update and checkpoint boundaries

Persistent G starts unready. Every named gradient still needs an explicit successful accumulation, including zero contributions; unreferenced parameters are never silently marked ready. Whole update preflight rejects any missing entry. ParamScale and mixed-AD scaling do not move between graph and coordinator routes.

After all copies, update retains P/G/M/V/EMA disjointness and existing AdamW/EMA/clipping formulas. Parameters may mutate only after all reduction/update preflight succeeds. Step increments and readiness clears only after the complete update and mutable-state checks succeed. Snapshot/restore preserve full P/M/V/EMA shapes and owned-host semantics. Shared workspace is ephemeral and excluded from checkpoint payloads; restore must create/admit fresh shared storage before execution.

## Isolated verification before integration

An approved implementation must add tests without model/native execution first:

1. Exact inventory/byte admission, current and released extents, one-byte boundaries, required-capability rejection before allocation, partial-allocation/upload retention.
2. Prefix checked bounds/expired owner/foreign driver/range alias rejection; stale-tail poison; views cannot free parents or extend shape.
3. Multiple host/dense/compact branches, repeated/unused/zero-scale parameter readiness, primary-only roots, scale-once, signed zero, deterministic order and independent F64 reference error distributions under existing gates.
4. Fault injection at each validation, upload, accumulation, copy-back, completion wait, cancellation, consumer close, update, state check and snapshot. Confirm retained owners, empty failed publication, no retry/update/checkpoint and earlier gradient preservation.
5. Compare global max/fixed-max/recomputed norm, clipped multistep AdamW moments/updates/EMA and checkpoint/resume with independently generated pinned references. Report error distributions and performance overhead rather than relaxing thresholds.
6. Full-step PTX/Vulkan mocks, weight endpoints, masked/repeated-token/empty-voice rows and late failure; legacy scopes/options hashes/public constructors remain unchanged.

Use a distinct versioned experimental shared-workspace scope and options-hash kind, with driver capability checks before allocation. Do not reinterpret existing replay identities even if snapshot wire format stays the same. Same-scope resume must verify the new policy; cross-scope restore must reject or require an explicitly reviewed migration. No default or public builder route changes.

## Conditional generic/mock-only approval

`@default` conditionally accepted a bounded generic/mock-only implementation in `/workspace/exports/pocket-sharedworkspace-design-review-20261002.md`. The optional capability must add pure norm-stat validation for **both gradient-stat passes and all P/M/V/EMA state checks**, in addition to merge/clip/update/copy validation. Admit the entire post-update check plan before the first optimiser write; numerical postconditions still require execution. Missing capability rejects before allocation. A separate design judge independently identified this state-check preflight omission.

Prefix views must carry enforced operation-generation validity, owning driver/context/allocation and checked physical range. Workspace reuse invalidates earlier views even while the parent remains live. Zero prefixes are read-only; scratch/branch/stats write permissions are operation-scoped. Views cannot close parents or enter retained tape roots. Every driver operation resolves and validates the view before access; stale views, foreign identity, parent expiration and reentrant use reject. Preserve mode0 maximum, mode2 correction/counts and A/B ping-pong with odd/even and short-after-long tests.

Approved work is optional generic capability/checked byte plan/opt-in constructor and host/fake accumulation/copy-back/recomputed-stat/fault/numerical tests only, with stable G and complete preflight. Return the bounded diff and evidence for review before adapters or CLI. Existing constructors/scopes and runtime ceilings/guards/downloads remain unchanged. No native/PTX/Vulkan adapter or execution/model/audio/reset/listening/publication approval is granted.

## Design evidence and decision

`TestTrainingReleasedDeviceSharedWorkspaceProposal` and `TestTrainingReleasedDeviceProposalCoexistence` calculate these numbers using synthetic inventories and existing graph planners. They call no proposed driver API or resident constructor. Logs are under `/workspace/exports/pocket-gpu-training-releasedresources-20261002/proposal-arithmetic.log` and `proposal-coexistence.log`. Final isolated audit/proposal window: 2026-10-02 16:49:41--16:51:20 UTC. Three repetitions passed 24 top-level +12 subtests; race passed 8+4; affected audit/CLI checks passed 10+4, all with zero failures/skips. Vet, host build, ARM64/RISC-V test compilation, layout and diff checks passed. Runtime logic is unchanged; no native or model execution occurred.

The resident-alone arithmetic fits the existing maximum, but combined native/host headroom is unqualified and copy/view capabilities are missing. The conditional decision permits only the bounded generic/mock-only slice above. Native adapters and CLI integration require another decision. Current released-size CLI admission continues to reject; all guards, ceilings and execution/publication holds remain.
