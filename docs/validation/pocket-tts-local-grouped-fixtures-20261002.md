# Pocket-local grouped fixture coordinator

The corrected test-only coordinator rejects overlap between every writable G and every live P before freezing or execution. Focused alias checks pass; prior grouped component and ownership results remain preserved. It owns mock P/G arrays and uses the accepted Pocket scalar adapter plus public pure tape planners. It has no production resident, optimiser, checkpoint or native connection.

The original local-grouped submission required this admission correction despite its passing tests. Review `/workspace/exports/pocket-local-grouped-source-review-20261002.md`; corrected evidence `/workspace/exports/pocket-local-grouped-alias-corrected-20261002`. The original bundle and logs remain unchanged.

Parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Owner `@go-pherence`; scoped authorisation `/workspace/exports/pocket-interpreter-adapter-correction-decision-20261002.md`. New files:

- `model/pockettts/training_grouped_local_test.go`
- `model/pockettts/training_grouped_local_invariants_test.go`
- `model/pockettts/training_grouped_local_oracles_test.go`
- `model/pockettts/training_grouped_local_alias_test.go`

The accepted three-file adapter, original scalar interpreter, backend grouped prototype, seven-file generic seal and separate later-phase source remain unchanged. The staged inference index remains eight files / 469 insertions. Production files, scopes, ceilings, guards, downloads and CLI routes did not change.

## Admission and execution

`pocketGroupLocal.run` fences the mock state before admission and keeps it fenced through plan/construct/dispatch/copy/close/publication callbacks. Nested work and mock update/snapshot/close/accumulate probes reject. Execution remains caller-serialised; the probes enforce exclusion only and implement no update or checkpoint bridge.

Initial admission bounds tensors/commands/outputs/bindings to 4,096, shapes to ranks 1–3, operands to 16, op names to 64 bytes, parameter names to 512 bytes, selected result roots to 16, groups to 128 and names per group to eight. The source/effective tape, frozen operation payload, planned slots and output payload retain their separate 1 MiB limits; command work is at most 65,536. The metadata override can only lower the 1 MiB ceiling.

Before copying/freezing, the coordinator reserves full-source metadata, 144 bytes per tensor for effective headers/validation inventory, planner scratch, binding/group/result-policy metadata and closure output headers. It validates the complete original source, exact declared G roots, typed result geometry, mock P/G inventory/extent/finiteness, complete G–G and all-writable-G × all-live-P disjointness, and the combined frozen payload. P, source constants/seeds, groups, bindings, compact descriptors and result-policy arrays receive operation-private copies. Stable G headers remain separate from frozen P. Frozen P protects this invocation's reads but cannot prevent overlapping G copyback from changing live weights. Admission rejects that overlap directly; it never restores P after mutation. Adjacent disjoint views of one allocation remain valid.

Effective source headers point to the privately copied P values and clear `External` only for admission. That complete source validates before pruning. No callback runs before freezing. Every group's selected gradient union must cover the full declared binding name set exactly once; all surviving P reads are remapped independently of selected G names, including cross-parameter reads.

The owned closure view declares selected parameter roots so even P-independent gradients preserve their binding. Original source outputs remain unchanged. Each retained group plan validates resident/compact provenance, exact root union, remapped original/effective source, declared typed results, supported opcode tuples, adapter metadata/output ceilings and `PlanZeroTapeStorage`. A full-source upper bound is reserved before allocating each pruned plan. No execution starts until all plans pass; there is no one-group replanning contract.

Each constructed mock owner retains its plan and frozen borrow references. The unchanged adapter copies per-group constants and P, allocates reusable/zero-pooled slots, executes, and returns owned arrays. Results/witnesses and all selected gradients/index arrays validate before G changes. Dense or compact accumulation applies `prior + scale*gradient` once into a scratch candidate, then completes a synchronous copy into stable G. The owner closes before the next group constructs. The closed owner releases plan/output references; uncertain owners are retained exactly once.

Any failure with a constructed owner or earlier execution progress is terminal, including a later nil-owner construction failure. Committed G has no rollback; a within-group copy fault can leave earlier names committed and later names untouched. This fixture proves a partial group commit, not a partial write inside one Go `copy` or a partial native transfer. A successful close followed by cancellation remains terminal but never resurrects the closed owner. Candidate metrics/input results publish as owned copies only after all groups close; failure at the final publication boundary returns no result.

## Typed policies and exact inventories

Component policies have distinct role/extent inventories. Flow roots include the complete raw time-adjoint boundary; the established decoder converts those adjoints into time VJPs. Transformer uses output/sequence VJP. FlowLM uses sequence, full-prefix Z, voice VJP and normalized-audio VJP; the pinned audio Z comparison selects the suffix.

Exact fixture-derived names/extents remain those accepted in `oracle-inventories.json`: flow 34 per each of two mixed cases, transformer 12, FlowLM 19 minus exactly `out_eos.weight` and `out_eos.bias` = 17. Each fixture checks schema 1 and upstream `0acce6b2f390150267557770d2098c5caa9a18ac`. Expected arrays come from the pinned PyTorch fixtures, never from the adapter.

Two deterministic partitions/orders use ascending or descending names, chunks of at most eight, at scales 0 / .25 / 1 with nonzero prior G. The separate compact fixture uses both `a→b` and `b→a` group orders, repeated tokens and nonzero inactive-token G. Its independently specified dense reference preserves untouched tokens.

The later graph policy validates original ordered seed/metric/voice/normalized/tail roots before pruning, freezes all decoder inputs and invokes the real `decodeStepResultsBoundary` on every group. It supports active nonzero latent-sized tails and rejects a corrupted tail after an earlier group committed. Selected IDs remain the accepted five/six root map; dense and compact conversion preserve them.

## Independent numerical results

Every value/VJP and parameter comparison keeps its accepted family tolerance. Parameter G references use the exact F32 `prior + scale*fixtureGradient` expression. Error distributions aggregate across both partitions/orders and all scales.

| Independent scope | Comparisons | Mean absolute | RMS | Maximum absolute | Existing tolerance |
|---|---:|---:|---:|---:|---|
| Flow P-gradient accumulation | 4,584 | 1.7097306e-9 | 1.04805202e-8 | 1.97440386e-7 | 1e-5 |
| Flow values/VJPs | 132 | 7.17740146e-9 | 1.33362109e-8 | 3.44589353e-8 | Value 3e-6; tangent 5e-6; input/condition/time 8e-6 |
| Transformer P-gradient accumulation | 864 | 5.22492074e-9 | 1.26796481e-8 | 8.94069672e-8 | 1e-5 |
| Transformer values/VJPs | 144 | 7.65236715e-8 | 1.01944732e-7 | 2.38418579e-7 | Output 4e-6; input 8e-6 |
| FlowLM subset P-gradient accumulation | 1,140 | 2.06457263e-7 | 7.87769482e-7 | 1.04904175e-5 | 2e-5 |
| FlowLM values/VJPs | 324 | 5.38097488e-8 | 1.32955698e-7 | 4.47034836e-7 | Sequence 2e-6; Z 5e-6; input 1.5e-5 |
| Compact independent reference | 54 | 0 | 0 | 0 | 2e-6 |

FlowLM supplies the pinned EOS input contribution `DZ + DEOS*EOS.Weight`; EOS parameter gradients are excluded by name. The larger FlowLM maximum uses its parameter tolerance, not a universal relaxed gate.

Later dense/compact graphs cover voice frames 0/1 and active/inactive final rows in both partitions/orders: **3,776 schedule comparisons**, zero mean/RMS/max error under `2e-6`. Row condition/target/logvar derivatives, raw row loss/square metrics and forward EOS scalar are supplied. Weighting/FlowLM/EOS gradients and input/tail outputs are computed from those boundaries. Earlier flow-head G is absent. This diagnostic does not establish a complete grouped training step or independent full-step oracle.

## Ownership invariant map

| Invariant | Pocket-local regression |
|---|---|
| Original/effective semantic admission, no repair by rebinding | `MalformedAndLifetime/effective-gather`, `/invalid-original-gather`; original accepted adapter tests retained |
| All-group admission before first construct | `Preflight` empty/missing/duplicate/foreign/oversize/late-duplicate/tuple/result/output/nested/compact cases and `plan-late` |
| Reserve metadata before freeze; cumulative later reservation | `Preflight/metadata-beforefreeze`, `/metadata-late` (one plan admitted; no execution) |
| Frozen cross-group P/constants/bindings/groups/results | `FreezeCrossP` mutates caller/live inputs at plan callbacks; A.G still reads frozen B.P |
| Exact component inventories and unchanged numerical gates | `FlowOracle`, `TransformerOracle`, `FlowLMOracle` plus accepted inventory helpers |
| Stable G and scale once/prior contents | Component oracles, `FreezeCrossP`, `CompactPrior` |
| All writable G disjoint from every live P before freeze/execution | `LivePGAliasAdmission/same-name`, `/cross-name`, `/partial-same-name`, `/partial-cross-name`; unchanged P/G/backing/readiness, inactive nonterminal state, zero freeze/plan/construct/dispatch/callback/publication |
| Successful G accumulation preserves live P | `LivePGAliasAdmission/disjoint`; adjacent cap-limited P/G views accepted, stable G, two completed groups and one publication |
| Early/late uncertain owner retention; nil owner after progress | `FailureFences` construct/nil/constructed/run/witness/download/lookup/actual/copy/close at groups 0/1 |
| No resurrected closed owner | `FailureFences/closed*`, `Cancellation/closed`, `/publish` |
| Constructed-but-not-run and callback cancellation | `Cancellation` construct/constructed/run/dispatch/dispatched/witness/download/lookup/actual/copy/close/closed/publish |
| Within-group partial group commit, no rollback/publication | `PartialCopyAndWitness/partial-copy`; earlier parameter copy completes, later parameter untouched; no partial native-transfer qualification |
| Nonfinite witness/G rejects before G changes | `PartialCopyAndWitness/nonfinite`, `WorkAndOverflow/accumulate-overflow` |
| Actual typed tail fails after earlier committed group | `TypedTailLateFailure` (eight first-group G names committed; nil result; one retained owner) |
| One owner at a time; copy+close before construct | `FreezeCrossP` and component/later ledgers `MaxLive=1`, constructed=closed |
| Reentry exclusion across full operation | `Reentry` plan/construct/constructed/run/dispatch/witness/download/lookup/actual/copy/close/closed/publish |
| Final publication fault discards candidate | `FailureFences/publish*`, `Cancellation/publish` |
| Owned output lifetime after source mutation/mock close | `MalformedAndLifetime` success suffix |
| Malformed policy/context/state never panic/execute | `MalformedAndLifetime` nil state/context/ready, invalid geometry/mask/policy and related cases |

The current retained-plan metadata limit rejects the larger synthetic work case before the 65,536-command branch: **975,400 reserved + 161,230 requested > 1,048,576 bytes**, after two plans / 1,028 charged commands, zero construction/dispatch. The command-limit branch is present but unexercised under that binding constraint. Limits were not raised.

## Host resource ledger

All numbers below are explicit allocation accounting or conservative reservations. No peak heap, RSS or device physical peak was measured.

Operation payload charges every frozen source `Values` slice and copied typed-policy array, plus a separate frozen P snapshot. Pruned plans retain metadata and references to that payload; they do not recopy P. `MetadataReservation` includes scratch/effective-validation allowances and `RetainedPlanMetadata` is its plan component. Per-group adapter ledger counts its additional constants/P snapshots, slots (including displaced borrowed slots), and returned arrays. Candidate, publication and accumulation scratch copies are separately charged. Rebound bytes are a subset of slots, never subtracted.

| Representative ascending grouping | Plans / commands | Operation payload / frozen P | Retained plan metadata / total reservation | Cumulative adapter constants / P copies / slots / returns | Candidate / published / accumulation scratch |
|---|---:|---:|---:|---:|---:|
| Flow mixed case, each | 5 / 1,893 | 2,316 / 1,528 | 532,326 / 953,732 | 9,620 / 7,640 / 17,032 / 3,556 | 100 / 100 / 1,528 |
| Transformer | 2 / 226 | 2,404 / 576 | 69,340 / 206,495 | 4,808 / 1,152 / 5,840 / 1,344 | 96 / 96 / 576 |
| FlowLM | 3 / 484 | 6,304 / 760 | 152,763 / 359,394 | 18,624 / 2,280 / 25,436 / 2,408 | 296 / 296 / 760 |
| Later no voice, inactive, dense | 4 / 959 | 5,390 / 880 | 352,272 / 717,311 | 17,824 / 3,124 / 22,508 / 2,000 | 60 / 60 / 880 |
| Later no voice, inactive, compact | 4 / 959 | 5,418 / 880 | 353,532 / 720,212 | 17,852 / 3,124 / 22,472 / 1,964 | 60 / 60 / 880 |
| Later voice, active, compact | 4 / 1,007 | 6,118 / 880 | 362,792 / 739,420 | 20,796 / 3,220 / 21,744 / 1,996 | 68 / 68 / 880 |

`host-ledgers.json` records all 13 logged representative rows, cumulative adapter metadata and maximum per-group allocations. Cumulative metadata across closed groups can exceed 1 MiB; the operation reservation and each individual adapter invocation obey their respective ceilings. GC release timing is unmeasured. The mock state contains only tiny P/G, with no M/V/EMA or resident backend. These ledgers do not qualify the released 86 MB device group plan or multi-gigabyte host-state ledger.

## Saved gates and limitations

Alias-correction window **21:07:23–21:08:22 UTC**, offline GPUs disabled, `GOMAXPROCS=2 -p=1 -parallel=1`:

| Correction gate | Top-level / subtests | Fail / skip |
|---|---:|---:|
| Ten `LivePGAliasAdmission` repetitions | 10 / 50 | 0 / 0 |
| New alias race | 1 / 5 | 0 / 0 |
| Alias excerpt | 1 / 5 | 0 / 0 |
| Existing `MalformedAndLifetime/G-alias` negative | 1 / 1 | 0 / 0 |

All four rejected overlap cases preserve original P/G backing contents and readiness, return an error/nil result, leave the state inactive/nonterminal and execute no callbacks, freeze, plans, construction, adapter dispatch or publication. The positive case preserves P while accumulating the independent tiny reference into stable G. Vet/format/diff pass. The fix adds only the live-P overlap loop before operation copies; it allocates no new execution payload and changes no arithmetic, planner, resource limit or ledger formula. Component/later numerical and broad gates were not rerun. Previous statistics and host ledgers below remain saved evidence from the original disjoint fixtures, not new measurements.

Main new-only window **20:53:38–20:53:58 UTC**, offline GPUs disabled, `GOMAXPROCS=2 -p=1 -parallel=1`:

| Gate | Top-level / subtests | Fail / skip |
|---|---:|---:|
| Ten repetitions | 140 / 880 | 0 / 0 |
| Race | 14 / 88 | 0 / 0 |
| Excerpt | 14 / 88 | 0 / 0 |

After that window, the compact reference fixture was tightened from one same-group partition to two explicit compact/dense group orders. Only that changed test reran at **20:55:12–20:55:30 UTC**: repeat 10 / 0, race 1 / 0, excerpt 1 / 0; zero fail/skip, 54 independent comparisons still zero error. A nil-ready-map admission guard and negative were added at **20:56:09–20:56:33 UTC**: only `MalformedAndLifetime` reran, repeat 10 / 170 and race 1 / 17, zero fail/skip. Necessary vet/diff passed after both changes; unchanged numerical gates were not repeated.

`gate.sh`, JSONL, excerpts, counts and follow-up commands preserve these separate source states. Draft compile/fixture errors, flow raw-boundary ordering, FlowLM prefix seed/suffix mistakes and the initially mislabelled work case remain archived. The final source uses the actual established builder APIs. Two bounded source-review delegates timed out without findings; neither counts as completed independent review. `@default` retains source/next-scope approval.

A new-only coverage run reports 9.3% for production package statements. Go excludes `_test.go` implementations from that profile, so it cannot measure this coordinator's statement coverage. The explicit invariant table replaces no numerical or native gate; test-only statement coverage remains unmeasured.

Original evidence `/workspace/exports/pocket-local-grouped-20261002`; gate `/workspace/tmp/pocket-local-grouped-gate.sh`. Correction evidence `/workspace/exports/pocket-local-grouped-alias-corrected-20261002`; gate `/workspace/tmp/pocket-local-grouped-alias-gate.sh`. Return the corrected source/evidence before further implementation. Subsequent smallest full-step/row-boundary proof work is read-only planning only; optimiser, checkpoint and production resident bridges remain unauthorised. Full-model/native completion, physical memory, FTZ/performance, model/voice/audio quality and publication remain open; all execution/recovery/listening holds are unchanged.
