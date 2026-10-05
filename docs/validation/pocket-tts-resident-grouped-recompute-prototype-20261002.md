# Test-only grouped recomputation coordinator

The isolated generic/mock coordinator holds one owner fence across every gradient group and publishes results only after all groups close. Focused numerical and failure tests pass. No callable runtime API, production routing, native adapter or released-model execution has changed.

Parent: `a90aec30567d485784f2e04d1803c543c9f2817f`. `@default` accepted the corrected synthetic ledger and authorised this test-only slice in `/workspace/exports/pocket-working-set-corrected-design-review-20261002.md`. The implementation lives entirely in `backends/training/grouped_recompute_prototype_test.go`; every helper is test-only. The seven sealed generic runtime/test files remain byte-identical for source provenance. Inference remains staged eight paths / 469 insertions. No commit or publication is established.

## Owned operation, complete admission

`groupedProbe.run` first calls `ResidentState.admit`, then sets the existing private borrow guard until final publication returns. Planning, construction, witness validation, gradient lookup, actual-buffer checks, accumulation/copy, close, between-group transitions and publication callbacks all run under that one guard. The prototype does not loop public per-tape entrypoints.

Request cardinalities are now admitted before validation walks or defensive copies: source outputs ≤4,096, requested results 2–16, tensors/commands/bindings ≤4,096, groups ≤128 and group size ≤eight. Nested shapes, operands, operation/name strings are bounded before copying. Full F32 source remains limited to 1 MiB, cumulative commands to 65,536, and conservative copied/planner metadata charges to 1 MiB. The exact union of selected names must equal the supplied graph binding names once each; it does not imply unrelated resident parameters are ready. Empty, foreign, duplicate, missing and oversized groups reject before tape construction. Full original parameter bindings remain separate from selected group gradient names.

Only vector mode 0 and compact embedding mode 3 are supported in this prototype. Unsupported/stateful commands reject. The proof is intentionally limited to those pure mock fixtures; the complete Pocket TTS combined graph needs broader opcode and numerical coverage before any production implementation.

Tape metadata, constant values, group names, binding maps and result IDs receive private copies before callbacks. Every current P source is copied into a private read-only mock snapshot. This prevents even direct test-hook mutation of original constants, group arrays, bindings or live P from changing the recomputed operation. These are host defensive copies, not zero-transfer device borrows: host payload includes the source `Values` sum, `4*sum(parameter extents)`, copied command/shape/group metadata and pruned group metadata. No benchmark or released host-peak reduction is asserted. Native implementation would instead need enforced immutable live parameter borrows through the same owner fence.

## Plan admission versus actual buffers

Every group's closure, remap, protected-parameter inventory, compact descriptor shape and zero/lifetime storage plan is validated before any tape constructor runs. Before allocating the next closure, the prototype reserves a conservative full-source-sized plan charge, remap/slot scratch, roots, results and binding/name map entries against the cumulative metadata byte ceiling. Charges include 256 bytes per tensor/command record, eight per nested ID, string payloads and conservative planner/map headers; they are bounds, not measured allocator bytes. All original live P sources are rebound when their reads survive pruning, including unselected gradient names. Planning callbacks cannot mutate frozen inputs.

Pure plan validation supplies topology/extents and work bounds. It does not pretend to validate future allocations. After a group executes, the prototype checks actual result count/extent/finite values and a zero scalar witness before touching G. It checks actual gradient buffers for extent/finiteness, compact index structure, and invokes the existing complete shared accumulation/copy preflight against real mock buffers. A post-execution admission error is terminal even when its validator is pure.

This prototype does not prove immutable native plans, physical kernel availability or released complete-gradient admission. All-group pure descriptors are restricted to the supported mock operations and existing validators; native buffer/kernel contracts require a separate reviewed capability.

## Execution and publication

Cancellation is checked before every constructor, after successful construction before run, after successful close before another group, and after stage callbacks before underlying construct/run/download/lookup/close dispatch. A post-close cancellation clears the completed owner first, retains no already-closed tape and starts no next group. A constructed but not executed owner remains retained terminally if cancellation occurs before run.

Each group runs synchronously, downloads and validates its owned selected results, checks that recomputed results stay within the unchanged `2e-6` gate, gathers selected gradient/index buffers, accumulates once at the caller scale, completes exact copy-back into stable G, then completes tape close before the next group exists. The mock tracks live owners and permits at most one group at a time.

The first decoded result is copied; later results are checked, never added into it. Metrics and selected input outputs publish once after every group has closed. The last publication callback remains fenced and may still reject/cancel the operation terminally. Results remain independently owned after tape/state closure.

Any execution, lookup, witness, actual-buffer admission, accumulation, copy-back, cancellation, close or late constructor failure fences the entire sequence. Even a nil-owner constructor failure between groups marks state terminal after earlier progress. Existing G values stay as committed; there is no rollback claim. An uncertain tape is retained; previously closed completed tapes are not resurrected. No later group starts, and no partial metrics/result escapes. A first nil-owner construction failure before any execution remains nonmutating/nonterminal.

Tests explicitly assert early witness/actual rejection before any G copy, failures in the second of three groups with the third never started, partial failure within a two-name group, earlier committed values, terminal retry/update/snapshot/close fences and last-group publication rejection. Callback reentry rejects nested grouped work and public accumulation/update/snapshot/close during all stages, including shared accumulation/copy hooks.

## Numerical evidence

| Check | Comparisons | Mean absolute | RMS | Maximum | Unchanged gate |
|---|---:|---:|---:|---:|---:|
| Independent F64 compact/dense gradients; weights 0 / .25 / 1, fresh and nonzero prior G | 60 | 6.95387522e-9 | 2.59881833e-8 | 1.1920929e-7 | 2e-6 absolute |
| Pinned independent PyTorch P/EMA update | 1,212 | 1.0052768e-9 | 6.26213903e-9 | 1.1920929e-7 | 4e-6 absolute |
| Three clipped updates versus established host P/M/V/EMA schedule | 120 | 0 | 0 | 0 | 2e-6 absolute |

F64 global norms use the existing `3e-7` relative-style bound. Repeated/unique compact tokens and valid prefixes 1/3 have independently derived masked scatter sums. Selected two-element result expectations are `[3,5]`; metric `.75` and witness zero publish once. Signed-zero/subnormal mock arithmetic is covered; exact copy bits remain the separate transport contract, so arithmetic sign-of-zero equality is not a generated-output gate.

Pinned fixture: `model/pockettts/testdata/training_step_pytorch.json`, schema 1, upstream revision `0acce6b2f390150267557770d2098c5caa9a18ac`. Its immutable named gradient arrays test group coordination/update against independently supplied expected P/EMA. They do not test recomputing a complete backward graph. The mock opcode tests derive gradients independently with F64/reference fixture arithmetic. Empty-voice/full-model/mixed-AD and released grouped backward remain open; no assets were fetched or executed.

## Narrow audit follow-ups

- Later-copy assertions now reject wrong lengths and nonfinite values before unchanged `2e-6` comparison; small length/NaN negative cases exercise the predicate.
- Empty-text and 256-frame source cases enforce exact `invalid Pocket TTS FlowLM training boundary` and `sequence tape byte ceiling` errors. The 1,025-token/voice probes enforce `step backbone rows`.
- The storage-design command comparison is labelled consistently: **792 pruned monolithic commands**, **5,560 cumulative pruned group commands**, **799 original full-source commands**. These are planned work counts, not latency.

The follow-up checks ran only changed copy/phase/sensitivity tests: three top-level and two subtest passes, no failures/skips. Logs are in `followups/`; this delegated run did not record a separate UTC window. The prototype gate below has explicit timestamps. The separate later-phase test and the seven-file runtime seal stayed unchanged.

## Focused prototype gates

Window **2026-10-02 19:03:30–19:03:38 UTC**. Go offline; PTX/Vulkan execution disabled; `GOMAXPROCS=2`, `-p=1 -parallel=1`.

| Gate | Top-level passes | Subtests | Failures / skips |
|---|---:|---:|---:|
| Ten prototype repetitions | 100 | 230 | 0 / 0 |
| Prototype race | 10 | 23 | 0 / 0 |

Targeted vet, format and diff checks pass. Untouched broad/numerical/resource gates were not repeated. Coverage profile is archived; test-only functions are not counted by Go's production coverage instrumenter. The production branches exercised by this mock cannot establish native execution coverage.

Initial fixture/API errors were corrected: wrong finite-helper name, empty compact map admission, dense-only event numbering, nil-owner cancellation expectation and the reference fixture's changing weight-dependent dense gradient. A clipped norm mismatch was traced to using the initial dense gradient after weights changed; the independent reference now derives it from current reference P each step. Initial failed logs remain archived.

A bounded read-only judge found no concrete coordinator/failure/reentry defect in the reviewed path, while listing proof gaps. Explicit compact-index postvalidation and within-group partial-copy/early-execution tests were then added before final gates. No rollback atomicity is claimed for a multi-name group. Full Pocket TTS opcode/result/numerical/native qualification remains outside this prototype.

## Independent correction review

`@default` identified missing post-close/pre-dispatch cancellation checkpoints and metadata copies before adequate bounds in attachment 3922. That bundle is preserved as superseded evidence for these contracts. The test-only corrections do not change the sealed runtime validator.

New tests cancel from pre-close, successful post-close notification, construct, successful post-construct notification and run callbacks, using constant-gradient groups with empty pruned parameter bindings. They assert exact constructor/run/copy/live/retention counts, earlier G values, no second G readiness and no publication. After successful prior close, no owner is resurrected; pre-close or post-construction cancellation retains exactly one uncertain owner.

Oversized 17-result and 4,097 repeated-output requests, nested shape/operand limits and a small cumulative metadata limit reject before freeze or before the next closure, with copy counters and plan callbacks proving the boundary. Pure rejection leaves state nonterminal/clean. A cross-group fixture has A's gradient read frozen B.P plus constant 3; the planning callback adversarially mutates live B to 77 and source constant to 99, while A's expected G remains **3.5** from B.P=0.5. This establishes all-P/selected-G separation directly rather than mutating an unread parameter.

Correction-only gates **2026-10-02 19:26:02–19:26:07 UTC**: **30 top-level + 100 subtest repeated passes**, **3 + 10 race passes**, zero failures/skips; targeted vet/diff and runtime seal checks passed. Untouched numerical, phase and broad gates were not repeated. Initial cross-fixture expected 4.2 was corrected to 3.5 after inspecting `residentFixture` B.P=0.5; failed draft evidence is archived. Two delegated design/review attempts timed out and supply no approval.

Correction evidence: `/workspace/exports/pocket-grouped-recompute-corrected-20261002`. The permitted subsequent [full-Pocket harness design](pocket-tts-grouped-full-harness-design-20261002.md) is read-only source/oracle/host-cost planning. Complete grouped backward execution remains missing.

Earlier evidence: `/workspace/exports/pocket-grouped-recompute-prototype-20261002`. Gate: `/workspace/tmp/pocket-grouped-probe-gate.sh`. Return this source and new evidence to `@default` before a callable runtime/shared-production API proposal. [Resource policy](pocket-tts-resident-backward-storage-design-20261002.md) remains design-only; native adapters, CLI routing, ceiling/guard/download changes, assets/audio/recovery/listening/publication stay held.
