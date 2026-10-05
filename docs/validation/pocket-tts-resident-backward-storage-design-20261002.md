# Bounded backward recomputation and resource-policy proposal

Sequential gradient-group recomputation can reduce the synthetic dominant backward owned plan without changing its full-source admission charge. At the tested released 32-frame / 32-token / no-voice geometry, nine groups of at most eight parameter gradients reduce the largest planned owned payload from 338,370,500 to 86,273,984 bytes. The pruned monolithic plan has 792 commands; the nine pruned groups total 5,560 commands. The full expanded source has 799 commands. No released-shape grouped execution, production runtime implementation, performance or native qualification has occurred. A separate small test-only mock coordinator is documented in [the prototype report](pocket-tts-resident-grouped-recompute-prototype-20261002.md).

Parent `a90aec30567d485784f2e04d1803c543c9f2817f`. `@default` authorised offline sensitivity and bounded storage/policy design after audit-only corrections. The [working-set ledger](pocket-tts-resident-working-set-20261002.md) records those corrections. Existing constructors, full-source/all-live ceilings, backend scopes, guards, downloads, checkpoints and staged inference are unchanged. Review this proposal before any runtime implementation.

## Synthetic sensitivities

`TestTrainingWorkingSetSensitivityAudit` uses released YAML extents and synthetic zero arrays. It builds and prunes graphs, never executes a model or driver. Unique tokens are distinct vocabulary IDs; repeated tokens use ID 0. Masked suffix rows contain zero sample times and derivative inputs. Graph shape remains the physical frame count.

| Frames / text / voice | Valid rows | Token pattern | Combined compact full-source bytes | Owned-plan bytes | Shared resident + owned plan |
|---|---:|---|---:|---:|---:|
| 7 / 3 / 2 | 7 | Unique | 1,010,120,280 | 309,263,976 | 2,148,766,400 |
| 7 / 3 / 2 | 3 | Unique | 1,010,120,280 | 309,264,004 | 2,148,766,428 |
| 32 / 32 / 0 | 32 | Unique | 1,229,822,240 | 338,370,500 | 2,177,872,924 |
| 32 / 32 / 0 | 32 | Repeated | 1,229,822,240 | 338,370,372 | 2,177,872,796 |
| 32 / 32 / 0 | 8 | Unique | 1,229,822,240 | 338,370,628 | 2,177,873,052 |
| 32 / 8 / 32 | 32 | Repeated | 1,266,486,760 | 343,012,100 | 2,182,514,524 |
| 128 / 32 / 8 | 16 | Unique | 1,714,130,536 | 406,984,196 | 2,246,486,620 |

Partial masks preserve the complete source inventory. Changes in immutable zero sharing affect the synthetic owned plan slightly; they do not remove padded frame extents or native work. Compact embedding roots still have physical-row output extents even when active tokens repeat. Mask and unique/repeated pairs now assert equal full-source charges.

32 frames / 1,025 tokens / no voice and 32 frames / one token / 1,025 voice frames reject at `step backbone rows`, before combined tape construction or compact rewrite. These are sampled topology boundaries, not a complete admissible-geometry envelope. The retained-row audit separately rejects its 256-frame parent at `sequence tape byte ceiling` and its 128-frame distillation coexistence at the 2 GiB all-live tape comparison. A storage proposal does not remove those logical rejections.

## Existing dominant lifetime

The combined reduction/weighting/EOS/backbone backward has many parameter-gradient outputs. `runBoundTape` gathers every output and compact index before whole-pass accumulation (`backends/training/resident_tape.go`). `PlanTapeStorage` pins declared output reshape families through the end of the tape, so later gradient production cannot reuse their slots (`backends/training/tape_storage.go`). Each resident G then receives a completed scaled accumulation and exact copy-back from shared scratch (`shared_workspace.go:243–316`).

Consequently full parameter gradients remain live together. Compact embedding removes one dense vocabulary VJP, but dense transformer gradients still dominate. Persistently writing graph outputs into G, or releasing them before last use without a completion-aware contract, would weaken current admission/lifetime boundaries.

## Proposed bounded generic/mock slice

Use a private experimental runner which recomputes the existing pure combined graph sequentially for disjoint parameter-gradient groups. It should preserve all original tensor IDs and arithmetic commands, with full-source and compact-binding admission before selecting group closures. No new native kernel is required by this design.

1. Validate the complete source, all bindings, compact index provenance, all requested result roots and the union of gradient names. Every expected gradient appears exactly once across groups. Reject missing/duplicate/foreign group entries before allocation or upload. Keep original public outputs/builders unchanged.
2. Sort names deterministically and form bounded groups. Eight gradients per group is the arithmetic probe, not a tuned default or measured optimal batch size. Bound the group count and total recomputation work as part of the private API.
3. For each group, prune only its requested gradients/index roots plus the numerical admission/result witnesses needed for unchanged arithmetic and output guards. All original parameter reads continue to borrow the same resident P, including names whose gradients are computed in another group. Do not charge these as new tape-owned parameter copies.
4. Admit every group's kernel topology and owned-slot plan before the first group runs. Use the same immutable step inputs, current weights, seeds and masks for every group; prohibit updates/snapshot/reentry until all groups have completed. The current driver only preflights against real buffers, so pure plan preflight and post-execution buffer/range checks must be clearly separated in a future mock contract.
5. Complete a group's execution, validate its output/admission witnesses, then accumulate its unique gradient roots with the existing scale-once compact/dense operations and completed copy-back. G remains stable. No parameter update or metrics/replay/checkpoint publication occurs between groups.
6. Close the completed group before creating the next. Full-source charge `Lcombined` remains the original expanded graph charge for each live consumer. A single-group owned plan never replaces it as the admission gate; groups must not coexist accidentally.
7. Any execution, accumulation, copy-back, cancellation or close failure is terminal. Retain the uncertain tape/state owners, preserve earlier committed gradients without claiming rollback, reject retries/update/snapshot/checkpoint/replay/metrics and do not start another group. Pure preflight failure before any execution must remain nonmutating.
8. Publish one coherent result only after all groups are complete and all owners close. Recomputed metrics/condition/input results must not be summed repeatedly. A later group failure must fence an earlier decoded result.

The arithmetic probe retains reduced numerical result roots in every group. A runtime proposal could select one result-producing group after full admission, but that has not been measured here. Do not subtract that possible saving from the recorded plan.

## Recalculated probe

At 32 / 32 / 0 unique tokens, the expanded full-source charge is **1,229,822,240 bytes**. Sorted gradient names produce nine groups. Every planner invocation protects all original resident parameter IDs and subtracts borrowed slots once; protecting only the selected group's parameter map would incorrectly charge other live resident reads as tape-owned.

| Quantity | Monolithic compact | Sequential group proposal |
|---|---:|---:|
| Largest owned plan | 338,370,500 | 86,273,984 |
| Shared resident plus largest plan | 2,177,872,924 | 1,925,776,408 |
| Full-source charge per live tape | 1,229,822,240 | 1,229,822,240 |
| Pruned planned commands | 792 | 5,560 cumulative |
| Original full-source commands | 799 | 799 unchanged source |

Repeated tokens produce 86,273,856 owned / 1,925,776,280 combined bytes, with the same source and command counts. The test asserts reduced peak plans, higher planned work and unchanged full-source admission. These planner results leave **221,707,240 bytes** below a hypothetical combined 2 GiB payload comparison for unique tokens, without budgeting backend overhead. No arbitrary reserve percentage is justified. Host state/gradient payload of about 4.69 GB is unchanged.

## Alternatives not selected

- Direct output binding to persistent G risks changing readiness, scale-once behaviour and terminal failure surfaces. Repeated gradient contributions also need the old G preserved while the tape writes. It requires a new write/accumulation capability and independent numerical review.
- Last-use streaming gradient consumption could avoid recomputation, but needs execution checkpoints, output-retirement semantics, source/witness ordering and completed copy-back before slot reuse. Existing whole-tape synchronous APIs do not establish those boundaries.
- Raising resident/tape ceilings or removing host guards would not reduce physical work and is outside the authorised scope.

Sequential recomputation offers a smaller generic ownership contract using current pure graph semantics, at substantial repeated work. Source/model purity and deterministic parameter reads must be enforced; a future stateful operation would make this proposal invalid without additional rules.

## Numerical and failure qualification before runtime approval

A future mock implementation must compare every named gradient, clipped multistep P/M/V/EMA update, metrics and selected input gradients against the existing ungrouped host schedule and pinned independent PyTorch/F64 references with unchanged tolerances. Test objective weights 0 / .25 / 1, repeated/unique tokens, empty voice, masks, negative zero/subnormal transport, and group boundaries spanning compact/dense parameters.

Keep exact comparisons only for graph metadata identity, group inventory and raw copy completion contracts. Generated arithmetic needs error distributions and independent tolerances. Grouped command pruning/reordering must preserve the actual dependency schedule; a changed hash is not numerical evidence.

Fault tests must cover early/late group preflight, lookup, dispatch, output validation, accumulation, copy-back, cancellation and close. Confirm earlier G contents, terminal retention and publication fencing after a later failure. No broad previously completed gate should be repeated without a source change that requires it.

## Resource-policy design

Keep current resident-only and all-live full-source limits unchanged. A proposed total device budget is separate from either limit and must include measured/declared backend allocation inventory, peak live group-owned plans, pipeline/module/staging scratch and alignment. If backend overhead is unknown, report a lower bound and reject a feasibility claim; do not assign an invented percentage or infer success from one-frame headroom.

Host policy needs live/boundary/snapshot payload, persistent gradients/scratch, frozen/nontrainable models, corpus buffers, metadata and GC coexistence. Streaming checkpoint encoding saves an encoded model-sized allocation but does not remove three live P/M/V/EMA sets during capture. Each budget should state geometry, backend, phase, inventory version and failure retention policy.

`@default` authorised an isolated test-only generic/mock prototype. Its source and focused evidence must return for review before any callable runtime implementation; native adapters and CLI routing are outside that permission. Native physical peak, completion/FTZ behaviour, transfer-inclusive performance and voice acceptance require separate authorisation and evidence.
