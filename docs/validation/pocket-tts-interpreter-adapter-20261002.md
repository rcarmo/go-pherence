# Additive Pocket test interpreter adapter

The corrected test-only adapter validates the frozen values it will execute, enforces exact component gradient inventories and asserts ordered later-graph result roles. Targeted repeat/race checks and independent flow, transformer and FlowLM comparisons pass. The adapter is not connected to grouped resident state.

This report supersedes attachment 3926's provisional admission, oracle-completeness and root-role submission. Its original source, report and logs remain preserved. The correction decision is `/workspace/exports/pocket-interpreter-adapter-source-review-20261002.md`; corrected evidence is `/workspace/exports/pocket-interpreter-adapter-corrected-20261002`.

Parent `a90aec30567d485784f2e04d1803c543c9f2817f`. `@default` accepted attachment 3924 and permitted only this additive test-only adapter/contract slice; decision `/workspace/exports/pocket-grouped-recompute-correction-review-20261002.md`. New files:

- `model/pockettts/training_interpreter_adapter_test.go`
- `model/pockettts/training_interpreter_adapter_contract_test.go`
- `model/pockettts/training_interpreter_adapter_corrections_test.go`

The original `training_flow_tape_test.go`, seven-file generic runtime/test seal, corrected backend grouped prototype and separate later-phase audit remain unchanged. Staged inference remains eight paths / 469 insertions. No production API, native adapter, CLI, resource ceiling, guard/download, checkpoint identity or numerical tolerance changed.

## Adapter admission and execution

`runPocketAdapter` returns errors normally. Its scalar switch was copied into the new test file from `interpretFlowTapeStorageBound`; the old interpreter is intact. It never catches `testing.T.Fatal`, panic or `Goexit` to emulate an error-returning interpreter.

Preflight rejects unsupported exact operation/mode/kind tuples before allocation. The supported family table is explicit:

| Operations | Modes | Kinds |
|---|---|---|
| `eos-admission`, `reshape` | 0 | 0 |
| `lsd-loss`, `eos`, `time-embedding`, `attention-forward`, `vector` | 0–2 | 0 |
| `embedding`, `linear-backward`, `row` | 0–3 | 0 |
| `relu`, `gelu`, `rope`, `linear-forward`, `silu-forward`, `silu-backward` | 0–1 | 0 |
| `attention-backward` | 0–4 | 0 |
| `mod-forward` | 0–1 | 0–1 |
| `mod-backward` | 0–3 | 0–1 |
| `norm-forward` | 0–2 | 0–1 |
| `norm-backward` | 0–3 | 0–1, excluding kind 1 / mode 3 |

`Tape.Validate` first checks the complete original source's shape, input/producer ownership and command semantics, including dead inputs. Original-source validity cannot be repaired by rebinding. All bindings then pass read-only-source, extent and finite-value admission, including dead external declarations; protected IDs must be valid.

The adapter freezes the source and copies bound arrays. A bounded second tensor-header inventory replaces each bound tensor's `Values` with that private copy and sets `External=false`. `effective.Validate` checks these exact execution values before slot planning/allocation, callbacks or dispatch. This preserves semantic admission for gather IDs, packed indices, EOS labels, LSD logvar/cutoff and time constants. Production validation and coordinator limits are unchanged.

Correction regressions reject vocabulary-2 ID `[2]`, fractional gather IDs, malformed dense/compact indices, EOS label replacements, LSD logvar/cutoff and time-frequency replacements. Each asserts an error, no returned output, zero slot payload/dispatch/callbacks and unchanged caller buffers. Valid ordinary-P and gather-ID replacements execute; an invalid original gather source remains rejected even with a valid replacement.

Cardinality limits are 4,096 tensors/commands/outputs/bindings, 1–3 shape entries, at most 16 operands, and operation names at most 64 bytes. Source F32 payload, conservative metadata and returned output payload each have a separate **1 MiB** test ceiling. Metadata includes nested shape/operand IDs, output IDs, map headers, value records and conservative validation/planner scratch, plus 128 bytes per effective tensor header and 16 bytes per tensor for the second validation's inventory scratch. These limits are local to the adapter and do not raise the corrected coordinator's limits.

Source metadata/constants and borrowed arrays receive private host copies before callbacks. The storage planner preserves protected positive-zero constants and borrowed reshape families. The adapter deliberately allocates all planned slots first, then rebinds entire borrowed slot families; its host ledger includes the allocated slots subsequently displaced by borrows. Non-reuse reshape creates another copy and charges it. Reused output destinations are cleared before the unchanged `+=` scalar reductions; reshape does not clear borrowed storage.

Cancellation is checked at entry, before every command, after the pre-command callback before dispatch, and after the post-command callback. The returned error includes no partial output, and no later command dispatches. Returned outputs are independently copied and finite-checked, so they outlive slots and never alias caller inputs or one another.

## Computed component oracle map

Every fixture checks schema 1 and `UpstreamCommit = 0acce6b2f390150267557770d2098c5caa9a18ac`. These are local test arrays; no model checkpoint or audio assets were loaded.

| Fixture / computed scope | Comparisons | Mean absolute error | RMS | Maximum | Existing per-family gates |
|---|---:|---:|---:|---:|---|
| `flow_head_backward_pytorch.json`: two mixed cases, reuse/nonreuse, value/tangent/input/condition/time and all flow parameter gradients | 1,572 | 4.23488273e-9 | 1.74892247e-8 | 1.97440386e-7 | Value 3e-6; tangent 5e-6; input/condition/time 8e-6; params 1e-5 |
| `transformer_backward_pytorch.json`: output, input VJP and all transformer gradients, borrowed P/zero storage | 168 | 2.16104603e-8 | 4.32779788e-8 | 2.38418579e-7 | Output 4e-6; input 8e-6; params 1e-5 |
| `flowlm_training_pytorch.json`: sequence/audio Z, voice/audio VJP and conditioner/backbone gradients | 244 | 3.97786622e-7 | 1.16976589e-6 | 1.04904175e-5 | Sequence 2e-6; Z 5e-6; inputs 1.5e-5; params 2e-5 |

The FlowLM tape excludes two EOS parameter roots. Its seed explicitly supplies `DZ + DEOS*EOS.Weight` from the pinned independent fixture; the adapter computes conditioner/backbone VJPs from that boundary. This is not a claim that this tape computes EOS parameter gradients. The mixed flow test does compute its flow-head parameter gradients from source inputs and seeds rather than supplying final gradient arrays.

Before comparison, `adapterOracleInventory` requires the exact translated fixture name set, unique parameter and gradient IDs, exactly one declared output occurrence per gradient, matching expected extents and finite oracle values. `adapterComputedInventory` requires the same exact computed name set, extents and finite values. Negative regressions reject missing/foreign names, duplicate or undeclared roots, duplicate output occurrences, wrong extents and missing computed results.

| Component | Exact expected inventory |
|---|---|
| Flow | 34 names per mixed case; fixture names prefixed with `flow_lm.flow_net.` |
| Transformer | 12 names; prefix `flow_lm.`, with `final.*` translated to `out_norm.*` |
| FlowLM backbone subset | 19 fixture names minus exactly `out_eos.weight` and `out_eos.bias`; both exclusions must exist; remaining 17 names prefixed with `flow_lm.` |

All comparisons check expected length, finite actual/expected values and unchanged tolerance. Mean/RMS/max errors are aggregated across each component. The FlowLM maximum `1.04904175e-5` is within the applicable existing input/parameter gates (`1.5e-5` / `2e-5`); sequence and Z retain their tighter gates. This aggregate maximum does not introduce a universal tolerance.

## Later combined graph contracts and boundaries

Four tiny cases cover voice frames 0/1 and active/inactive final audio rows. `stepBackboneFixture` supplies independently specified row derivatives/metrics for graph-contract testing; its hidden forward array is obtained through the established interpreter. The adapter builds the actual combined reduction/weighting/EOS/backbone tape, appends real reduced step-result roots, then records dense and compact manifests. Compact conversion has complete original binding admission before appending compact index/gradient roots. Its original result IDs remain stable.

`contracts-oracles.log` records every emitted `op/mode/kind` count, ordered result ID/extent/producer tuple and named parameter/gradient ID/extent before and after compact rewrite. The later dense graph emits 34 distinct tuples from attention, embedding, EOS, GELU, projection/norm, ReLU, reshape/RoPE, row and vector families; compact adds `embedding/3/0`. Other supported flow/time/modulation modes are exercised by component fixtures or remain admission-only table entries; support is not inferred from nearby modes.

`adapterStepRootRoles` asserts five ordered selected roots without voice and six with voice: diagonal logvar, distillation logvar, five metrics, optional voice VJP, normalized audio VJP and latent-sized direct tail. Every root must be declared, distinct and correctly sized. The logvar vector writers must read their respective admitted raw seed at input 1; the normalized/metrics writers must have the expected vector mode. The tail writer must trace to `Tape.Outputs[1]` with the final-row offset and latent extent. Tail discovery uses writer semantics because compact conversion appends gradient/index roots after it. Dense and compact contracts both assert these roles; a negative dense case swaps the same-shaped logvar roots and must reject.

The exact selected root map is unchanged by compact conversion and by the active/inactive tail choice:

| Voice frames | Ordered role → ID (elements) |
|---|---|
| 0 | Diagonal logvar → 12 (2); distillation logvar → 15 (2); metrics → 95 (5); normalized audio VJP → 480 (4); direct tail → 482 (2) |
| 1 | Diagonal logvar → 12 (2); distillation logvar → 15 (2); metrics → 95 (5); voice VJP → 478 (2); normalized audio VJP → 493 (4); direct tail → 495 (2) |

Typed result validation calls the real `decodeStepResultsBoundary`. The last selected root is a **latent-sized direct-tail vector**; a scalar-zero sentinel fails its extent contract. Its values must exactly match the final normalized-input gradient row; zero is required only when that row is inactive. Active cases deliberately produce nonzero tails. Negative checks reject tail mismatch, nonfinite metrics and overlapping output arrays. They do not weaken the real source-disjointness/mask/metric rules.

| Later combined quantity | Computed here | Supplied boundary / missing phase |
|---|---|---|
| Scaled diagonal/distillation logvar seeds | Reduction and weighting seed packing computed | Raw per-row logvar seeds supplied in `StepReductionInputs` |
| Weighted row metrics/EOS total | Reduction computed | Row loss/square metrics and forward EOS scalar supplied |
| Weighting MLP gradients | Computed from samples/raw seeds | Earlier loss inputs supplied |
| FlowLM conditioner/transformer/EOS gradients | Computed from combined condition/EOS seeds | Earlier diagonal/distillation condition derivatives supplied |
| Final audio/voice input gradients and tail | Computed from backbone plus direct-target boundary | Earlier direct row target derivative supplied |
| Flow-head parameter gradients | **Not in this later graph** | Earlier diagonal/distillation backward phase remains separate |

Dense/compact selected results agree within `2e-6`, with zero observed diagnostic error across 15 or 17 values per case. That is an ungrouped schedule diagnostic, not an independent full-step oracle. No claim compares every full-step PyTorch gradient against this later graph alone. Complete training-step grouping remains open until the earlier row contributions, flow gradients and their provenance are mapped/executed as well.

## Host payload and metadata ledger

The ledger reports allocated payload, not RSS or native physical peak:

`FrozenConstants + BorrowCopies + SlotPayload + ReturnedPayload`

`ReboundSlotPayload` is a subset of `SlotPayload`, reported separately and never subtracted from host allocations. Slot payload in non-reuse mode includes discarded fresh reshape destinations and the subsequent reshape copies. Metadata is a conservative charge, separate from payload; no Go allocator/GC overhead is measured.

| Tiny case | Frozen constants | Borrow snapshots | Allocated slots/output scratch | Owned returned arrays | Rebound slots included | Metadata charge |
|---|---:|---:|---:|---:|---:|---:|
| Flow mixed, no reuse | 2,316 | 0 | 16,272 | 1,628 | 0 | 378,574 |
| Flow mixed, zero/reuse | 2,316 | 0 | 4,460 | 1,628 | 0 | 378,574 |
| Transformer, borrowed P | 2,404 | 576 | 3,256 | 672 | 576 | 126,638 |
| Borrowed reshape regression | 8 | 8 | 24 | 16 | 8 | 3,428 |
| Later dense, inactive tail / no voice | 5,148 | 0 | 6,692 | 988 | 0 | 332,272 |
| Later compact, inactive tail / no voice | 5,176 | 0 | 6,752 | 1,048 | 0 | 333,545 |
| FlowLM backbone subset | 6,304 | 0 | 9,028 | 1,056 | 0 | 187,001 |

The remaining active/voice cases and FlowLM payload are recorded in the evidence log. Graph builders may share source backing before freezing; the ledger charges every copied `Values` slice. The tiny adapter has no resident/G/M/V/EMA state and retains only one execution's slots. It cannot justify the released group's 86 MB device-owned plan or the approximately 4.69 GB host-state peak; those are separate ledgers.

## Test-only placement and bridge proposal

Backend `groupedProbe` helpers are private to `backends/training/*_test.go`. They cannot be imported into `model/pockettts` through `train`, and this adapter makes no such attempt.

A concrete next test-only integration proposal is to keep a new Pocket-local coordinator in `model/pockettts/*_test.go`, implementing only the previously reviewed owner state machine over a local mock state. It would consume this adapter through an error-returning test interface and existing public pure tape planners/descriptors. It would keep all P versus selected G maps, snapshots, witnesses and group result policies private to Pocket tests. Production internals remain untouched; no unsafe/linkname or test helpers exported from runtime packages.

A second option is a future approved test-support package with no production imports of its helpers. That would require explicit source/packaging review; no such package or export is added here. Duplication of the coordinator state machine would require the same cancellation/metadata/failure regressions in the Pocket package and a clear divergence check. A black-box loop over current public resident per-tape calls cannot provide one fence across all groups and is not proposed as equivalent.

Neither bridge is implemented in this slice. Connecting the adapter to grouped resident state or adding callable production APIs requires the next scoped decision.

## Correction gates and preserved attempts

Correction window **2026-10-02 20:06:27–20:07:58 UTC**. Offline Go; GPUs disabled; `GOMAXPROCS=2`, `-p=1 -parallel=1`. Only new corrections and admission/inventory-affected adapter checks ran.

| Gate | Top-level passes | Subtests | Fail / skip |
|---|---:|---:|---:|
| Ten affected adapter repetitions | 80 | 240 | 0 / 0 |
| Affected adapter race | 8 | 24 | 0 / 0 |
| Contract/oracle excerpt | 8 | 24 | 0 / 0 |

`counts.json` is recounted from saved JSONL. Targeted vet, format/diff and source-seal checks pass. The first correction gate reached vet after passing repeat/race/excerpt checks; vet rejected two unkeyed test binding literals. The keyed-literal fix changed no arithmetic; vet and seals then passed without repeating numerical gates. `vet-initial.log` preserves that failure.

`first-tests.log` records the draft logvar-role error (raw seed is vector input 1 via `branchScale`, not input 0). `second-tests.log` records the draft compact-tail positional error (compact roots append after the tail). Both assertions were corrected from the actual builders; `third-tests.log` passed. The final component errors above are unchanged. A bounded read-only judge inspected effective admission and the inventory/root-role helpers and found no concrete scoped blocker; full source/native approval still requires `@default`.

The original adapter window **19:46:55–19:47:33 UTC** recorded 70 + 90 repeated passes, 7 + 9 race passes and 7 + 9 excerpt passes, zero failures/skips. Those logs and draft fixture/API errors remain in `/workspace/exports/pocket-interpreter-adapter-20261002`. The original review found admission/completeness defects despite those passing counts; attachment 3926 is superseded by this correction.

Corrected evidence: `/workspace/exports/pocket-interpreter-adapter-corrected-20261002`; gate `/workspace/tmp/pocket-adapter-correction-gate.sh`. Original interpreter, grouped prototype, seven-file runtime/test seal and separate later-phase source match their preserved copies; the staged inference index remains eight paths / 469 insertions. No untouched broad/phase/completed numerical gates were rerun. Review the corrected adapter and exact maps before any grouped connection. Native completion/FTZ/physical memory/performance, real model/voice/audio quality and publication remain unqualified and held.
