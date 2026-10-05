# Pocket TTS working-set phases and defensive checks

The reviewed shared resident layout fits the existing resident-only 2 GiB ceiling, but retained tape admission depends on geometry. The synthetic 128-frame distillation phase exceeds the separate 2 GiB all-live tape ceiling. Native physical peak remains unmeasured.

Parent: `a90aec30567d485784f2e04d1803c543c9f2817f`. `@default` accepted the [generic/mock milestone](pocket-tts-resident-sharedworkspace-generic-20261002.md) with defensive gaps open, then authorised this isolated offline reconciliation and new reachable tests. The seven reviewed implementation/test files still match their seal. The corrected ledger and subsequent sensitivity/design slice add four test files and two reports; it changes no runtime constructor, backend, CLI, resource ceiling, guard, download, checkpoint identity or numerical tolerance. Training remains uncommitted; the unrelated inference index remains eight paths / 469 insertions.

## Four different quantities

- Resident admission charges five persistent P/G/M/V/EMA arrays per trainable parameter, three largest-parameter workspaces, largest statistics buffer and 48 scalar bytes: `R = 20*E + 12*M + 4*Smax + 48`.
- Full-source tape admission sums every declared F32 extent, including dead tensors, parameters borrowed from resident storage and external edge inputs. The root lease adds every live child's full-source charge. The resident ceiling and tape ceiling are separate; adding both logical charges double-counts borrowed arrays and is not the existing acceptance rule.
- Owned storage plans first prune exact declared/downloaded/gradient roots, then apply intra-tape last-consumer reuse and positive-zero pooling, and subtract each borrowed physical slot once. They deduplicate reshape families and resident/external borrows. They do not measure native memory.
- Host payload counts uniquely owned P/M/V/EMA sets, persistent gradient workspaces and transient snapshots separately. Metadata, Go allocator overhead, garbage not yet collected and backend overhead are additional quantities.

`ResidentTapeLease` charges `logicalBytes` and `root.liveBytes` before child allocation (`backends/training/resident_tape_child.go:124–139`); `validateConsumer` checks against remaining tape budget (`resident_tape_lease.go:219–237`). Successful close removes the charge; failure retains the tree (`resident_tape_lease.go:333–371`). No accounting subtraction weakens full-source or all-live admission.

Released synthetic inventory: `E=89,448,706`, largest parameter `M=4,194,304`, flow-head subset `F=9,758,752`. `R=1,839,502,424` bytes, leaving `307,981,224` bytes below 2 GiB. The default 256 MiB still rejects the resident layout. No production driver implements the shared capability, and selectable scopes continue to use their existing layout.

## Retained device phases

| Phase | Live owners | Logical charge | Owned plan payload |
|---|---|---|---|
| Forward backbone/EOS | Resident plus one retained parent | `R` separately; `Lparent` against tape ceiling | `R + Oparent` |
| Row condition | Parent plus one immutable condition child | `Lparent + Lcondition` | `R + Oparent + Ocondition` |
| Row forward | Parent, condition, three diagonal or six distillation roots | Previous charge plus `Lforward` | Previous payload plus `Oforward` |
| Scalar admission | Same owners plus temporary flag/witness consumer | Previous charge plus `Lguard` | Previous payload plus `Oguard` |
| Row backward/accumulation | Parent, condition, retained forward, backward consumer | `Lparent + Lcondition + Lforward + Lbackward` | `R + Oparent + Ocondition + Oforward + Obackward` |
| Reduction/backbone backward | Row children and parent closed; combined reduction/weighting/EOS/backbone backward is one later consumer | Complete combined dense source plus compact rewrite addition | Resident plus this later tape's owned plan, never summed with expired row children |
| Optimiser update | All step tapes closed | Resident state unchanged | Resident plus scalar diagnostics; host sync follows |
| Snapshot | Only resident state, host trainer/boundary/workspace and new host arrays | Resident admission unchanged | `R` plus host payload |
| Failed execution | Uncertain owners retained, all further publication/mutation fenced | Live charges retained | No rollback/release estimate |

Parent creation uses `BeginZeroPrunedTapeLease` and pins selected hidden, EOS loss, finite witness and EOS flag (`model/pockettts/training_forward_results.go:103–123`). The row vector is a separate read-only child (`training_device_handoff.go:13–34`). Forward children close before row results are published (`training_diagonal_device_inputs.go:216–224`, `training_distill_device_owner.go:224–228`). The condition then closes (`training_device_handoff.go:62–73`). Parent closure is required before later reduction/update and rejects surviving children (`training_flow_handoff.go:111–118`, `training_step.go:61–78`).

The guard child precedes backward rather than coexisting with it. The audit now calls the actual admission builders and planner: at hidden 1024 / latent 32, diagonal full logical charge is **17,988 bytes**, distillation **19,572 bytes**, and each owned plan is **4,248 bytes**. These include finite-witness reshapes, zero vectors, dots and scalar sums. The logged row peak uses the maximum of guard and backward phases, never their sum.

## Synthetic geometry ledger

`TestTrainingWorkingSetPhaseAudit` builds tapes and storage plans from `testdata/english-upstream.yaml`, with shared synthetic zero backing. It does not load a checkpoint, construct a trainer, allocate a native driver, run a graph or read audio. The fixture supplies complete parameter extents; named values are rebound as borrows in storage accounting. Source builders still admit full graphs before pruning.

All numbers below are bytes. `rows = 1 + frames + text tokens + voice frames`. Each row combines the retained parent, condition child, retained row forward and shape-only backward. The guard phase is included in the phase maximum.

| Frames / text / voice | Rows | Diagonal all-live logical | Distill all-live logical | Resident + diagonal owned plan | Resident + distill owned plan |
|---|---:|---:|---:|---:|---:|
| 1 / 1 / 0 | 3 | 1,206,883,044 | 1,444,903,164 | 1,879,167,308 | 1,879,174,056 |
| 7 / 3 / 2 | 13 | 1,248,208,380 | 1,486,228,500 | 1,880,192,572 | 1,880,199,320 |
| 32 / 8 / 4 | 45 | 1,385,011,748 | 1,623,031,868 | 1,883,925,500 | 1,883,932,248 |
| 128 / 32 / 8 | 169 | 1,969,050,148 | **2,207,070,268** | 1,901,150,268 | 1,901,157,016 |

Every row exceeds the default 256 MiB tape cap. The first three rows stay below the separate 2 GiB tape cap. At 128 frames diagonal stays below it, while distillation exceeds it by **59,586,620 bytes**, before omitted additions. The test now asserts the 128-frame acceptance/crossover and the 256-frame parent rejection; a review found that the first draft merely logged these cases.

At 256 / 32 / 8, the parent source builder rejects with `sequence tape byte ceiling` under 2 GiB, before storage planning. Empty text at 1 / 0 / 0 rejects the tested source builder with `invalid Pocket TTS FlowLM training boundary`; this is the builder's contract, not a claim that every production empty-text route rejects.

The corrected parent full-source/owned-plan pairs are respectively 968,856,080 / 297,804; 1,010,156,840 / 1,323,068; 1,146,857,808 / 5,055,996; 1,730,502,992 / 22,280,764. The condition owned plan is 4,100 bytes. The much smaller owned plans follow output pruning and borrowed parameter storage; full-source admission remains unchanged.

## Later phases and compact embedding

`TestTrainingWorkingSetLaterPhaseAudit` reconstructs the post-row combined reduction/weighting/EOS/backbone backward, applies the private reduced-result roots, then performs the existing compact embedding rewrite. Full dense source admission still precedes the rewrite; its appended compact roots increase the full-source charge. Pruning uses public download roots plus every parameter gradient and compact index root, matching the coordinator.

| Frames / text / voice | Combined dense full logical | Compact full logical | Dense owned plan | Compact owned plan | Resident + compact owned plan |
|---|---:|---:|---:|---:|---:|
| 1 / 1 / 0 | 968,845,696 | 968,849,808 | 320,317,424 | 303,937,536 | 2,143,439,960 |
| 7 / 3 / 2 | 1,010,107,952 | 1,010,120,280 | 325,635,648 | 309,263,976 | 2,148,766,400 |
| 32 / 8 / 4 | 1,145,371,812 | 1,145,404,680 | 343,105,024 | 326,753,892 | 2,166,256,316 |
| 128 / 32 / 8 | 1,713,999,076 | 1,714,130,536 | 423,105,152 | 406,983,684 | 2,246,486,108 |

Compact rewrite adds respectively 4,112 / 12,328 / 32,868 / 131,460 logical bytes and saves 16,379,888 / 16,371,672 / 16,351,132 / 16,121,468 owned-plan bytes with repeated token 0. These are synthetic compact-index costs; unique-token coverage and padding require their own geometry sensitivity.

All combined later tapes fit the separate 2 GiB source ceiling but exceed its 256 MiB default. Standalone reduction source/owned plans range 21,700 / 12,804 to 2,725,784 / 1,626,728 bytes; standalone reduction plus weighting range 59,028 / 19,432 to 4,083,404 / 1,910,632. The selected descendant composes them into backbone backward; do not add standalone and combined plans.

The later combined tape dominates the audited row-owned peak. At one frame, compact resident-plus-owned payload leaves only **4,043,688 bytes** below a hypothetical combined 2 GiB physical cap. At seven frames it already exceeds that comparison by **1,282,752 bytes**; at 32 frames by **18,772,668 bytes**; at 128 by **99,002,460 bytes**. This is a planning comparison, not a new runtime acceptance rule. Resident and tape ceilings remain separate. The tiny one-frame headroom does not establish native feasibility.

## Missing device quantities

These are phase lower bounds, not a complete-request physical peak. The ledger omits pipelines/modules, command buffers, descriptor/storage alignment, staging/copy scratch and allocator fragmentation. Host graph metadata and primitive scratch are separate. Native lifecycle/completion and FTZ behaviour have not been executed.

Full-source row forward/backward borrowing still includes parameter extents twice logically. The synthetic deduplicated plans avoid those physical copies, but leave no evidence about actual PTX/Vulkan allocation overhead. Intra-tape reuse does not establish cross-tape reuse. The 128-frame distillation logical rejection cannot be bypassed using its smaller owned plan without changing the admission contract, which is outside this scope.

## Host lifetimes

With EMA enabled and moments populated, one P/M/V/EMA set charges `16*E = 1,431,179,296` payload bytes. The live trainer's P arrays are the loaded trainable model arrays; do not count them again as separate immutable model weights. Frozen buffers and Mimi/conditioner/nontrainable model arrays are additional.

| Phase | Simultaneously owned host sets | Trainable state payload |
|---|---|---:|
| Prepared steady step | Live trainer plus backend completed boundary | `32*E = 2,862,358,592` |
| Initial resident construction, moments already populated | Live trainer, `t.State()` boundary and cloned `ResidentSnapshotFromFullState` input | Up to `48*E = 4,293,537,888` |
| Initial resident construction, no moments yet | Live P/EMA, boundary P/EMA, new complete snapshot with zero M/V | `32*E = 2,862,358,592`, plus temporary zero upload array |
| Device update sync | Live trainer, old backend boundary and newly downloaded resident snapshot | `48*E = 4,293,537,888` |
| Completed checkpoint capture | Live trainer, backend boundary and fresh `t.State()` checkpoint | `48*E = 4,293,537,888` |
| After transfer/boundary replacement | Live trainer plus new boundary; old arrays become collectable | `32*E`, excluding garbage not collected |

`FullTrainer.State` clones P/M/V/EMA (`full_training_state.go:128–145`, `training_state.go:390–402`). Initial conversion clones its arrays again (`training_resident_state.go:13–43`, `training_resident_backend.go:104–126`). `ResidentState.Snapshot` allocates all four host arrays and downloads them (`backends/training/resident.go:483–511`). Consuming conversion takes those arrays, then `loadResidentCompletedState` copies them into stable live arrays (`training_resident_state.go:62–81,90–110`, `training_resident_load.go:115–162`). Old `b.host` remains the template until `b.host=coherent` (`training_resident_backend.go:358–370`). Thus three complete sets really coexist during update sync; removing the second conversion clone did not remove live/boundary/snapshot coexistence.

The runner retains one reusable workspace per current frame shape (`training_runner_linux.go:169–186`). It owns FlowLM gradients and two full flow-head gradient sets (`training_workspace.go:145–151`, `flowlm_training.go:317–322`, `flow_head_backward.go:164–179`). One complete returned gradient inventory charges `4*E = 357,794,824`; the second flow-head scratch charges `4*F = 39,035,008`. During update sync these add **396,829,832 bytes**, producing **4,690,367,720 bytes** of host state/gradient payload before buffers/metadata. Weighting gradients are created separately for the returned step; checkpoint-only persistent gradient payload excludes those 2,241 elements once the step result is no longer referenced.

Checkpoint capture calls `r.trainer.State()` while resident/backend/workspace owners remain live (`training_runner_linux.go:300–324`). The binary writer streams F32 data with a 32 KiB conversion buffer through a 64 KiB buffered writer (`full_training_binary.go:263–279`, `training_runner_linux.go:328–340`); it does not need a second model-sized binary byte array. JSON helpers have different encoding costs and are outside this binary-runner estimate.

Further host costs: frozen buffers, corpus latent/token copies, flow scratch activations, reduction arrays, backward caches, time/frequency metadata, model tensors excluded from `fullParameterMap`, checkpoint publication buffers and not-yet-collected snapshots. Workspace replacement can temporarily retain old allocations until GC. A failed graph retains its device tree and current host references; no recoverable peak/rollback promise follows.

## Reachable defensive tests

`backends/training/shared_workspace_defensive_test.go` adds five tests / 20 subtests:

- Constructor rejection before allocation: nil context/driver, typed-nil driver, missing scale/global interfaces, cancellation, zero ceiling and invalid snapshot; nil allocations at first/middle/final boundaries retain the partial owner before uploads.
- Prefix/access: inactive operation, invalid extent, immutable zero, nil prefix result, nil token/context/driver, typed-nil identity, dead parent, closed/terminal state, generation overflow and cancelled scoped access.
- Later copy failure: first G copy succeeds, second fails partially; G identities remain stable, earlier readiness/dirty state persists, the failed gradient is unready, all execution/publication/retry/close paths are terminally fenced; executed tape source is retained. Added numerical checks assert host first G `[2,4]` and second `[6]`; compact first G `[3,4,0,0,6,8,0,0]`, second G `[3,-13]` with the untouched sentinel proving partial second copy. The unchanged arithmetic tolerance is `2e-6`.
- Complete late preflight: second host dense branch, later tape dense branch, and compact branch after earlier dense admission reject before any upload/accumulation/copy/readiness. Tape admission after execution retains the source terminally.
- Snapshot/close: callback reentry rejects accumulation/update/snapshot/close, snapshot download failure retains all owners, partial close failure rejects further release attempts.

The first defensive draft used the wrong binding fixture API; it was corrected before passing gates. The phase test first had API/fixture construction errors and missing active mask/vector placeholder/condition-root tracing; each was corrected against actual source. Those failed logs remain archived. No numerical threshold changed.

## New gates

New-only and one necessary affected-regression window: **2026-10-02 17:59:17–17:59:54 UTC**. Go offline, GPUs disabled, `GOMAXPROCS=2`, `-p=1 -parallel=1`.

| Gate | Top-level passes | Subtests | Failures / skips |
|---|---:|---:|---:|
| Ten defensive repetitions | 50 | 200 | 0 / 0 |
| Defensive race | 5 | 20 | 0 / 0 |
| Necessary two-package regression | 658 | 1,450 | 0 / 0 |
| Pure phase/planner audit | 1 | 0 | 0 / 0 |

Vet and diff checks passed. No production/build/kernel files changed. Defensive-only coverage reaches `view` 100%, constructor 95.1%, begin 90%; it does not rerun/requalify the established statistics/numerical paths (`sharedStats` is 0% in this intentionally focused new-test coverage).

The bounded two-file judge found no physical-peak overstatement but identified missing assertions for 128-frame admission/crossover and 256-frame rejection. Those assertions were added; the changed phase test alone passed **18:02:37–18:02:50 UTC**. Defensive/runtime files did not change, so completed defensive gates were not rerun.

The subsequent isolated later-phase test asserts all four source admissions, positive compact full-source additions, owned-plan reductions and the one-frame versus seven-plus-frame hypothetical combined-payload crossover. It passed with vet **18:09:50–18:10:04 UTC**; one new top-level test, zero failures/skips. Its initial fixture API/field errors were corrected before the passing run and retained in draft logs. A narrow supplemental review questioned reuse of pre-rewrite result IDs. `compactResidentEmbeddingTape` explicitly preserves original IDs and appends tensors/commands/outputs (`training_resident_embedding_compact.go:11–14,36–57`); new assertions now enforce original output ID prefixes and result extents before planner reuse. That changed test alone passed **18:12:09–18:12:22 UTC** with unchanged counts. No completed defensive/regression gates were repeated.

## Independent review corrections

`@default` rejected the retained-row table in attachment 3918 after inspecting actual builders and lifetime roots. Corrections are audit/test/report-only:

1. Capture hidden/loss/witness before appending EOS admission; then append its flag. Assert four distinct declared roots, extents `[frames*hidden,1,1,1]`, original ID preservation and ordered producer kinds/modes. This replaces the prior duplicated flag and omitted EOS loss.
2. Build and plan actual diagonal/distillation scalar admission graphs. The previous hardcoded 4,504 / 4,888 logical and 24 owned bytes were wrong; corrected numbers are above.
3. Redirect distillation condition reads and clear the unused source's `External` flag, exactly as production does. Preserve its full-source extent; validate resident bindings and the complete external inventory before pruning, with missing/duplicate/undeclared rejection assertions.
4. Add independently derived first committed and partial-second gradient content assertions to the later-copy fault test.

Prior source/report/logs and attachment 3918 remain archived. Its retained-row owned-plan figures are superseded by this table. Logical row peaks and independent later combined/host figures remain unchanged. The corrections passed **18:22:05–18:22:17 UTC**: changed copy test ten repetitions **10 top-level + 20 subtests**, race **1 + 2**, changed phase **1**; zero failures/skips, targeted vet/diff passed. Untouched broad gates and later-phase test were not repeated; the seven-file runtime seal remains intact.

## Subsequent sensitivity and design

The authorised offline sensitivity probe covers unique/repeated tokens, partial masks, larger voice prefixes and sampled topology rejections. It also calculates sequential parameter-gradient recomputation groups without executing them. Details and proposed numerical/failure contracts are in [the bounded backward storage design](pocket-tts-resident-backward-storage-design-20261002.md).

The new sensitivity test/vet passed **18:28:06–18:28:30 UTC**, one top-level test with no failures/skips. Full-source equality assertions were then added for unique/repeated and full/partial-mask pairs; that changed test alone passed **18:30:19–18:30:42 UTC**. No completed correction/defensive/broad gates were repeated. The design estimate at 32 frames / 32 unique tokens / no voice reduces peak owned plans from 338,370,500 to 86,273,984 bytes across nine groups, increasing cumulative pruned planned commands from 792 to 5,560; the expanded full source has 799 commands. No released-shape grouped runtime or performance claim follows.

`@default` accepted the corrected four-fix ledger and authorised a test-only grouped prototype plus narrow assertion/report follow-ups. Those follow-ups now reject wrong gradient lengths/NaN and enforce exact empty-text/256-frame/1,025-token/voice error categories. The [test-only prototype report](pocket-tts-resident-grouped-recompute-prototype-20261002.md) records new focused evidence and remaining numerical/native gaps. Resource policy stays design-only.

Correction evidence: `/workspace/exports/pocket-gpu-training-working-set-corrected-20261002`. Earlier evidence: `/workspace/exports/pocket-gpu-training-sharedworkspace-defensive-20261002`. Gate: `/workspace/tmp/pocket-sharedworkspace-defensive-gate.sh`. Return this ledger and the three-test-file source slice to `@default` before native adapter scope. Complete physical peak, unique-token/padding geometry sensitivity, released execution, numerical/native failure qualification and voice/listening/publication still require separate work and authorisation.
