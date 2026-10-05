# Vulkan sharedworkspace integration

The released shared-workspace runner now passes independent CPU parameter and EMA comparison, checkpoint, fresh resume, EMA export and recurrent inference gates for the 14-frame synthetic row. Grouped later gradients also pass an actual 32-frame update below the unchanged 2GiB native and 6GiB RSS limits. The earlier 1,293-outlier result below is superseded by corrected arithmetic. CLI host admission, joint-maximum numerical acceptance, efficiency against CPU, listening and Samantha adaptation are unfinished. Current staged joint-maximum native execution completes within resource caps but rejects two parameters at the unchanged 4e-6 gate; see the current-source status below.

## Allocation and grouped execution — 5 October 2026

Evidence: `/workspace/exports/pocket-allocation-fix-20261004/`. Every executed qualification has CPU/heap profiles and written conclusions, including failures. Exhaustive heap profiling timings are diagnostic.

| Actual native workload | Parameter maximum / failures above 4e-6 | Native peak bytes / allocations | Peak RSS bytes |
|---|---:|---:|---:|
| 14-frame grouped update/checkpoint | 1.8533319234848022e-7 / 0 across 89,449,730 | 1,888,061,824 / 970 | 5,185,736,704 |
| Fresh grouped resume, step two, EMA export | 2.1047890186309814e-7 / 0 across 89,449,730 | 1,888,061,824 / 971 | 5,216,419,840 |
| 32-frame/31-valid grouped synthetic update | 7.670460036024451e-7 / 0 across 89,449,730 | 1,897,911,392 / 969 | 5,535,748,096 |

EMA differences are at most 7.450580596923828e-9. Export storage matches EMA/buffers/frozen Mimi across 214 tensors and 109,502,146 values at 6e-8. CPU-checkpoint-backed prefill, flow, LSD, decoded PCM and two-frame session reset pass existing gates; PCM maximum difference is 7.222406566143036e-7 at 3e-5. Native counters return to zero after successful closure.

Runner pooling removes dense tape-only zero gradients. Synchronous leases retain frozen metadata without copied values; immediate child conversion avoids redundant constants while public/deferred conversions retain owned storage. Complete F32 finite scans classify exponent bits without dropping dead-source checks. Five alternating normal-profile fresh runs improve median qualification duration from 42.34s to 33.33s; this includes hashing/comparison/checkpoint and is not CPU/native CLI speed evidence.

Grouped later execution admits the original complete graph and all parameter/compact bindings before each group. Needed nonselected parameter sources remain bound. Sorted disjoint groups accumulate each gradient once. Failures fence the owner; no retry or publication follows. Shape-view canonicalisation, immutable zero-source interning and direct matrix gradient roots reduce the declared source, without changing arithmetic or source ceilings. Shared forward-only producers retain complete parameter inventory but omit reverse operations.

Read-only 375-frame/374-valid/13-voice/one-token admission now fits: forward source 1,214,958,960 bytes, later source 1,905,081,336 bytes; one-gradient groups need at most 301,874,192 owned bytes. Conservative resident-plus-tape payload is 2,141,397,096 bytes before padding, and host-payload forecast is 5,317,056,600 bytes before runtime/driver overhead. Separate SIMD step-one and fresh step-two references exist. Native execution was not launched because fresh idle checks found external WPE graphics activity. Joint maximum text/voice geometry is not covered.

A four-step released soak passes on the preceding monolithic revision with stable native residency; real submitted-work cancellation retains owners with one submission and no retry/drain/recovery. Grouping/source refinements need their final soak refresh. Affected suites, race selections, vet and host build have profiled evidence; full task acceptance still requires the open gates above.

### Connected staged route

Shared sequences above 128 total rows now use separately declared conditioner, original-index transformer layers, final norm, EOS and reduction/weight sources. The original monolithic joint-maximum source rejects correctly: 2,454,091,920 forward bytes and 4,526,442,308 reverse bytes at 375 target frames/62 voice frames/512 tokens. Each bounded layer source validates the whole original model first; selecting a layer never hides invalid omitted parameters. Private immutable boundaries and paired owner tickets fence input/model changes, incomplete updates and retries after failure.

Production staged preflight validates the complete graph and inventory before pruning, then charges resident state plus every executed boundary/gradient-root slot set. Borrowed parameter storage is charged in the resident owner; unaccounted external producers reject. The 950-row layer source is 592,844,628 bytes forward and 1,089,706,324 bytes reverse. With finite nonzero activation/seed inputs, maximum owned stage is 314,586,484 bytes; resident plus stage is 2,137,332,292 bytes, leaving 10,151,356 bytes below 2GiB before native alignment. The earlier zero-input forecast understated storage by one 950×1024 F32 slot. Production re-admits actual propagated sources before dispatch. Final norm and conditioner/EOS/weight stages fit separately. Host retained F32 is 2,252,491,024 bytes, with a 5,337,336,248-byte payload forecast before runtime/driver overhead.

Scoped forward placeholders reuse only existing admitted workspace zero extents, retaining distinct tensor IDs and logical source charges. This removes 95.51MiB from two-live producer storage and 192.02MiB from staged preparation allocation without adding retained scratch shapes. Computed graph-local zeros and unused-input bindings use existing validated operations; reduction association and public/legacy graph contracts remain unchanged.

A 129-frame tiny-model complete staged training call passes independent CPU metrics, input VJPs, AdamW parameters and EMA, both unclipped and with the runner's maxNorm=1 clipping. Per-stage forward/reverse failure injection, provenance mutation, resource-cap/dead-source rejection, affected full suites, race checks, vet and host build pass with profiles. These tests use the host interpreter. Ten normal and ten race checkpoint/fresh-resume repetitions also compare step-two parameters, EMA and Adam moments to CPU at unchanged gates. They do not establish released native resume or throughput acceptance.

### Current-source native status — 5 October 2026

Rui authorised GPU sharing for correctness/resource qualification, superseding the earlier idle-only restriction. Other graphics processes remain untouched; shared-GPU timings are not controlled efficiency evidence. Independent joint-maximum SIMD checkpoints remain pinned under `joint-max-cpu-reference` and `joint-max-cpu-resume`.

An exhaustive-profile run reached its 1,200-second deadline with retained in-flight ownership. Its failed evidence is preserved; no drain, reset or recovery followed. Profile-led fixed kernel-local descriptor scratch removes two per-dispatch slices, and plan recording no longer takes the address of a copied stage. Offline suites/race tests preserve descriptor, rebind and cancellation contracts.

The subsequent sampled-profile update completes, but one parameter exceeds 4e-6. Fixed-vector native reduction matches the CPU dot; an independently captured affected layer isolates activation drift. The forward-linear shader incorrectly used the CPU blocked K128 association above the CPU's 576-row dispatch ceiling. Correcting the general large-row shape path to the actual four-accumulator SGEMM association removes the selected layer's activation/gradient drift; fifteen native linear shapes pass existing CPU/F64 gates. CPU references, GELU and PTX remain unchanged.

The full joint run on the corrected source still rejects **two of 89,449,730 parameters**:

| Parameter | Index | Absolute error |
|---|---:|---:|
| `flow_lm.transformer.layers.1.linear2.weight` | 1,914,574 | 4.79724258184433e-6 |
| `flow_lm.transformer.layers.2.linear2.weight` | 120,489 | 4.122499376535416e-6 |

Native peak is **2,137,332,320 bytes / 929 allocations**; sampled peak RSS is **5,702,246,400 bytes**. These stay below 2GiB/4096/6GiB, with no cap hit, in-flight work, uncertain state or device loss at completed comparison. This is not complete acceptance: the unchanged numerical gate fences checkpoint publication, fresh native resume, EMA/export and ordinary closure proof. Remaining propagated-boundary drift must be fixed, not hidden by a larger tolerance. Evidence: `integrated-joint-correct-association`.

CLI host admission remains conservative and unchanged. Final-source native soak, matched efficiency, listening and Samantha comparisons remain open.

## Historical integration evidence

Implementation parent `a90aec30567d485784f2e04d1803c543c9f2817f`. The implementation is published as experimental, with the current-source failing gate recorded above. PTX source/kernels were not changed or executed in this slice. Rui cleared implementation/native qualification and required an audit before every run. Each offline/native invocation has a local pre-run record under `/workspace/exports/pocket-vulkan-sharedworkspace-20261003`; no other agent was contacted or delegated to. Staged inference stays eight files/469 insertions.

## Adapter and command

`VulkanSharedWorkspaceDriver` is a separate adapter. Existing `VulkanResidentDriver` constructors and routes stay unchanged. The new backend/replay identity is `vulkan-resident-sharedworkspace-hybrid-v1`; CLI name omits the trailing `-v1`. PTX sharedworkspace is unsupported and rejects explicitly.

The adapter supplies five persistent P/G/M/V/EMA buffers per parameter, shared scratch/branch/immutable zero/statistics, and three four-value globals through the existing generic coordinator. Persistent allocation wrappers bind an exact driver pointer. Prefixes validate operation-context pointer, generation, owner liveness, bounds and write permission on every transfer/driver resolution. They cannot close parents or enter retained tapes. Borrowed completed tape sources require the same kernel operator and cannot be written.

Full allocation inventory rejects physical aliases before initial uploads. Dense accumulation requires pairwise disjoint ranges and complete preflight; norm/update/compact/tape APIs preserve existing validation. Gradient copy-back uses bounded raw-byte copies through coherent mapped native storage under the Vulkan lane after prior compute completion/fence. There is no F32 arithmetic or separate host staging. This is CPU-mediated copy-back and must be included in performance/host-memory accounting. No GPU-only copy or speedup is established.

Copy failure/cancellation after progress is terminal. Earlier committed G may remain; ordinary retry/update/snapshot/close cannot treat uncertain ownership as a clean boundary. Existing kernel plans/leases retain uncertain resources. Scope inheritance selects existing device handoffs/compact embedding/state checks; new capability admission precedes host State copies and native allocation. Complete source admission, numerical guards and resident/live-source ceilings are unchanged.

## Tests and tiny native result

New offline tests cover exact raw-copy bits (signed zero/subnormal), nil/foreign/extent/physical alias/closed/cancel checks; missing/invalid access tokens; prefix close/retained-tape rejection; generic shared accumulation with short-after-long prefixes/repeated G; norm alias; failed submission/terminal retention/no retry; scope/capability/replay and host-budget overflow/boundary rejection. Mocks do not execute shader arithmetic.

Native `TestPocketNativeSharedFullStepPyTorch` runs only with `GO_PHERENCE_POCKET_NATIVE_FULLSTEP=vulkan-shared`. Three frames/two active,hidden4/latent2, same59/606 pinned upstream fixture. The exact adapter pointer remains unwrapped, because a wrapper would invalidate guarded driver identity. G stays device-side; independent pinned M/V and P/EMA qualify its update contribution.

| Quantity | Values | Maximum absolute error | Existing gate |
|---|---:|---:|---:|
| Final derivatives | 14 | 2.98023224e-8 | 5e-5 |
| P | 606 | 1.49011612e-8 | 4e-6 |
| EMA | 606 | 1.1920929e-7 | 4e-6 |
| M/V derived from independent G | 1,212 | 2.98023224e-8 | 4e-6 |

Five metrics also pass3e-5. Normal cleanup succeeds. Single diagnostic update interval857.20ms is correctness timing only. Previous Vulkan norm shader correction is unchanged; PTX untouched.

Focused new adapter/scope/CLI repeated gate:50 top-level+40 subtest passes; race5+4. Two directly affected legacy tests pass. Replay correction adds20 repeat/2race passes; host admission tests add20repeat/2race. Vet, formatting, host command build and diff checks pass. No broad or completed unrelated gates repeated.

## Released working set and command

Host-only graph audit uses the existing real Mimi-prepared synthetic tone corpus,14 physical/13 valid frames,13 reference frames,one token. The released model is pinned `fb0dc01b0d4d2e1c905b7a3e0676e3d9c96d5ae460e24e3ab94981805babf997`,219029196B. It builds but does not execute graphs; dummy boundaries serve topology/resource accounting only.

| Phase | Complete-source bytes | Planned owned physical payload |
|---|---:|---:|
| Transformer/EOS producer | 1,077,800,340 | 3,186,568 |
| Diagonal forward | 119,009,440 | 53,948 |
| Diagonal backward | 119,009,328 | 39,310,184 |
| Distillation forward | 238,018,112 | 57,420 |
| Distillation backward | 238,020,776 | 39,316,660 |
| Later expanded reduction/weight/EOS/backbone | 1,076,909,144 | 317,970,992 |

Complete live-source distillation coexistence is1,553,900,672B, within the existing2GiB all-live-source cap. Scalar admission child overhead is additional and small; runtime retains its checks. Shared residency is1,839,522,904B, independently below its2GiB cap. Maximum planned native buffer payload is2,157,493,896B, excluding alignment/driver/plan overhead. Existing policy has separate resident and live-source ceilings; there was no combined2GiB limit, cap lift or source-admission bypass.

The first command failed before state/tape allocation because the new replay scope was missing from `TrainingRunIdentity.validate`. The separate allowlist entry and test were added; original failure preserved. A newly audited fresh process then completed one update/checkpoint:

- Wall40.09s, including loading/preparation of training state and save.
- Peak RSS7,494,772KiB, about7.15GiB.
- Step1 loss.3115290403, flow.2960824668, EOS.1544656605.
- Coherent bundle1431217112B under new evidence directory.
- RTX3060/driver610.57.04; `nvidia-smi` sampled up to342MiB. Native coherent allocations may reside in host-visible system memory, so this is not total mapped/native memory. No device error/reset/recovery occurred.

This timing is slower than the prior CPU command7.37s and is a single end-to-end sample. Preparation/checkpoint/transfer overhead is not separated. No efficiency acceptance follows.

## Released numerical failure

An independent streaming binary decoder compares every P/fixed-buffer/M/V/EMA value against CPU step1 with identical model/corpus/PCG sampling/clipping/AdamW/EMA. Footer hashes verify file integrity only. Acceptance uses numerical4e-6, finite/nonnegativeV, exact names/extents/config/step; it records all above-gate counts.

| State | Values | Mean absolute error | RMS | Maximum absolute error | Above4e-6 |
|---|---:|---:|---:|---:|---:|
| P | 89,449,730 | 7.65998966e-10 | 4.03302937e-8 | 1.03178434e-4 | 1,293 |
| Fixed buffers | 320 | 0 | 0 | 0 | 0 |
| M/V | 178,899,460 | 1.09583181e-11 | 4.50053072e-11 | 3.38768587e-8 | 0 |
| EMA | 89,449,730 | 7.65020671e-13 | 4.76382544e-11 | 1.03143975e-7 | 0 |

Exact distributions are in `cpu-gpu-state-comparison.json`; the table retains the values from that saved output. No fullstep numerical acceptance is claimed. A second read-only diagnostic samples the first20 parameter violations, reads independent initial BF16 weights and reconstructs F32 AdamW from the saved CPU/Vulkan moments. Predicted and actual P agree at those20 points. Small differences in near-zero gradients/moments amplify through the first AdamW denominator. The remaining work is to trace released forward/backward rounding against an independent reference and downstream update effects; this observation does not justify widening any threshold.

## Host admission correction

The CLI previously charged CPU training plus checkpoint snapshot, excluding coherent native mapped residency/tapes and the backend's owned completed-state boundary. The measured RSS violated that estimate.

Only the new shared scope now also reserves4P bytes for the retained host boundary plus both separately capped native buffer sets. Total admission is CPU `ResidentUpperBytes` +9P bytes +2×device cap, leaving remaining bytes for the immutable model mapping. Sums are overflow checked. This is conservative and may reject shapes whose exact physical slots fit; it never hides mapped native RAM. Existing6GiB default and2GiB device caps are unchanged.

The corrected actual CLI rejects before model mapping/runtime/lock with:

```text
row0: sharedworkspace host/native mapped admission requires 9431175456 bytes before model mapping, ceiling=6442411896
```

The output directory stays empty. No native rerun follows the numerical failure/host-estimate defect.

## Offline host storage follow-up

Initial shared-state uploads now borrow only the backend's independent host State copy synchronously. Completed shared checkpoint writing similarly borrows the validated private host boundary. Public State/snapshot APIs retain deep-copy ownership.

A shared-only `ResidentState.SnapshotInto` now reuses independent host P/M/V/EMA arrays. Full destination inventory and pairwise host ranges are checked before download; model completion also checks all host/live arrays, including frozen buffers, before any overwrite. Partial transfer, late cancellation or later live-load failure is terminal. It cannot advance live state or publish a successful checkpoint. Other backend scopes retain their previous snapshot path.

Released inventory accounting removes4P=1,431,195,680 bytes of old/new completed-host snapshot overlap. The previous transition's payload lower bound was6,529,934,908B, already over6GiB before overhead. Reuse lowers the retained-state/gradient payload to5,098,739,228B, excluding mapping, tapes, source metadata, scratch, allocator/runtime and GC overhead. No revised RSS measurement or admission reduction follows; the conservative guard still rejects.

Five-sample isolated benchmarks: initial snapshot conversion787,952–787,953→263,664B/call (11→7allocations), completed snapshot at1,024values17,456→152B/call (6→4allocations). The larger mock benchmark timed out during its quadratic constructor alias scan; partial evidence is preserved and excluded. Transfer/native timing and whole-request efficiency remain unqualified.

Saved step-one moment analysis covers all89,449,730inferred clipped gradients: CPU norm1.0000000023184492, Vulkan norm1.0000000030790712, global ratio fit1.0000000007425043,186opposite non-zero signs. A positive global clipping factor alone cannot explain the drift. An offline released speaker projection check finds a CPU/sequential-F32 reduction-order difference before transformer execution (13,312outputs,max5.960464477539063e-8). The emulation does not execute Vulkan or establish propagation to the1,293P failures. Tolerances remain unchanged.

New focused repeat/race tests cover real completed-step checkpoint writing, ownership, writer failures, reused snapshot storage, host/live alias rejection, partial downloads, late cancellation and late load rejection. Scoped checkpoint helper coverage100%; SnapshotInto89.4%,completion85%,range helper80% remain below the95%admission-helper target. Focused vet/host CLI build/diff checks pass. Full report/audits/logs: `/workspace/exports/pocket-shared-hostcopies-20261003/report.md`. No GPU/PTX run, contact or delegation occurred.

## Offline sensitivity follow-up

The new test-only sensitivity controls reproduce saved CPU P/M/V at all1,293originalfailure locations. Speaker-only sequentialF32projection leaves1,184above4e-6; local-branch post-VJP objective scaling leaves1,209. Aggregate errors change little. Neither operation alone explains the released failure; no production arithmetic changed.

At663originaltransformerFC2weight failures, fixed CPU inputs/adjoints let an independent compensatedF64dot separate local reduction error from boundary drift. SequentialF32/F64alternative predictedPchanges are≤7.580965757369995e-7, below4e-6, and all26gradient-sign mismatches persist. A fixed-adjoint shaderGELUformula observation similarly changesP≤7.576309144496918e-7 and preserves those26sign mismatches. Larger drift reaches FC2through earlier activations/adjoints. Host formula emulation cannot qualify native transcendental arithmetic.

All29shippedtrainingSPIR-Vassets match current GLSL executable/layout/decorations after offline rebuild with the same target format. Initial27differences came fromSPIR-V1.0Uniform/BufferBlockversus1.3StorageBuffer/Blockencoding. Assets were not overwritten and no GPU/PTX ran. New tiny seams pass20+20repeat and2+2race checks; focused vet/hostCLIbuild/diff checks pass. Full new report and raw evidence: `/workspace/exports/pocket-shared-sensitivity-20261003/report.md`. Comparisons cover only originaloutliers; new failures elsewhere remain unscanned.

## Initial moments and host lifetime follow-up

Shared preparation now retains its independent step-zeroM/Vupload arrays for first completed download. This avoids another2P=715,597,840Ballocation while keeping `b.host.Adam` empty until successful completion. Full pre-download host/live alias admission, terminal download/failure ownership and public snapshot contracts are unchanged. Restored/legacy paths retain their existing storage. Tiny prepare/adoption/failure/allocation tests pass repeat/race; five-sample32,768value destination benchmark drops263,664–263,665→2,224B/call,7→6allocations. Its extra validation scan is slower; no latency claim follows.

The new unique retainedF32capacity ledger counts2,545,345,788Bfor initial host/model/workspace/corpus plus later source Values. Adding futureliveM/V,resident/plannednative tape buffers and immutable mapping forecasts5,637,466,720B,leaving804,984,224Bbelow6GiBbeforeexcludedallocator/runtime/metadata/deadGCoverhead. Host-onlyRSSincludeslazyzeropagesandnonativebuffers; admission/RSS remain unqualified and guard/caps unchanged.

A finaltransformernormadjoint source-formula intervention at fixed CPU forward boundary leaves1,222originalPfailures above4e-6;maximumCPU/emulatedboundarydifference1.1641532182693481e-9. It cannot alone explain released drift. Full report/audits/logs: `/workspace/exports/pocket-shared-norm-lifetimes-20261003/report.md`. NoGPU/PTX/network/agentsor tolerancechanges.

## Shared decoded restore follow-up

Shared positive-step runner restore now consumes the independently decoded checkpoint only after complete numerical/inventory/config/topology/alias validation and live copy. It retains those arrays as the independent backend host boundary, reuses live EMA, and avoids public parameter clones and another State copy at native preparation. Step-zero/nonshared/publicrestorepaths are unchanged. Pending restored boundaries validate live mutation before checkpoint/nativeallocation; equal-value checkpoint aliases explicitly reject.

CPU-only released restore attribution: cumulative workspace-phaseallocation5,595,700,144→3,448,782,616B,HeapSys4,579,786,752→3,271,196,672B,peakRSS4,482,432→2,818,752KiB. Finalretainedpayloadissimilar; repeatedisolatedbenchmarkB/call2,252,004–2,252,106→586,994–587,059. Nativebuffers/planswerenotallocated; these measurementsdonotqualifyreleasednativeadmissionorwholecommandefficiency. Caps/guard/GCpolicyunchanged.

Focused restore/storage/cancel/alias/replay/nextsnapshot/noEMAtests passrepeat/race. Scopedloadcore97.7%,hostseparation97.1%,SnapshotInto97.9%,allothernewsharedownershiphelpers100%,meeting95%changedhelperscope withoutunsafeoverflowfixtures. Fullreport/audits/logs: `/workspace/exports/pocket-shared-storage-resume-20261003/report.md`. NoGPU/PTX/network/agentsor tolerancechanges.

## EMA export inference follow-up

F32 EMA exports previously reloaded through `GetBF16`, truncating updated linear weights. `loadLinearF32` now preserves F32; existing BF16 storage and F16 conversion are unchanged. The pre-fix regression failed, then dtype/shape/malformed admission checks passed repeat/race. Loader coverage is92.3%, below95% with two immutable-file post-admission paths unexercised. Affected export/flow tests, vet, host CLI build and formatting/diff checks pass; staged inference remains eight files/469 insertions.

The existing CPU step-two synthetic-row export matches all109,502,146stored values across214tensors against checkpoint EMA/buffers and frozen Mimi. Direct checkpoint-model prefill/flow/LSD/decode and full-prefix/scalar two-frame generation/session reset pass unchanged numerical gates. Generated boundaries record zero difference for these inputs; shared model arithmetic limits independence. No output hash/byte acceptance, listening, speech adaptation or native qualification follows. Independent F64 affine controls fit a priori F32 rounding bounds.

CPU-only read/check/inference peakRSS is1,733,272KiB; the separate two-frame generator/session process peaks1,820,884KiB. These processes exclude training/export coexistence and native allocations. Caps/admission and the released1,293Pfailure are unchanged. NoGPU/PTX/network/agents, state update, new export/checkpoint or publication occurred. Full report/audits/logs: `/workspace/exports/pocket-export-inference-20261003/report.md`.

## Tape-only resident branch follow-up

Read-only released slot attribution finds302,379,912Bretained gradient outputs within317,970,992Bowned later tape; zero constants are only1,155,492B. Every shared-scope trainable gradient is tape-owned. An explicit tape-only resident constructor now omits the unused16,777,216Bhost-upload branch and rejects host accumulation before driver calls or mutation. Generic shared/legacy constructors keep their existing storage and host-upload contracts. Model preparation checks complete ownership before selecting tape-only mode.

Released resident payload falls1,839,522,904→1,822,745,688B; combined resident/later payload falls2,157,493,896→2,140,716,680B. Margin below2GiB is6,766,968Bbefore allocator/descriptor/plan overhead. Vulkan charges actual memory requirements including padding; no safe overhead bound or nativeRSSmeasurement was established. Caps and conservative CLI admission are unchanged; numerical failure still forbids native reruns.

New and directly affected tiny tape/constructor/snapshot/host-rejection/failure tests passrepeat/race. Model initial/restored preparation integration passesrepeat/race; new planner/constructor/core/ownership helpers have100%scoped coverage. Full portable training tests, affected vet, hostCLIbuild, formatting/diff and Vulkancompile-only pass. NoGPU/PTX/network/agentsor publication occurred. Full report/audits/logs: `/workspace/exports/pocket-shared-slot-ledger-20261003/report.md`.

## Attention and forward-linear sensitivity follow-up

At fixed CPU Q/K/V/probability/adjoint boundaries, serial shader-indexed attention scores/probabilities/output/dQ/dK/dV have zero measured differences across all six layers. Independent F64 attention adjoints differ≤7.450580596923828e-9. Control P/M/V reproduce saved CPU at all1,293originaloutliers. Host exp cannot qualify device transcendental arithmetic or upstream QKV parity.

First/final-row forward projection scans differ≤1.71661376953125e-5. A test-only transformer serial-F32linear forward propagation intervention leaves974originalPviolations; adding host forward-GELU formula leaves934. Opposite nonzero moment signs fall77→67→63 under explicit F32nonzero predicates. RMS Perror decreases9.0976e-6→8.6621e-6→8.5974e-6, but maximum error worsens1.0318e-4→1.0709e-4→1.1351e-4. No acceptance or production arithmetic change follows; new outliers elsewhere were not scanned.

Tiny contexts/masks/actualCPUAD/F64reference and explicit forward-GELU checks passrepeat/race. Test-only helpers are not instrumented by Go coverage; narrow production-package coverage2.6%has no helpercoverageclaim. Affected vet/hostCLIbuild/format/diff pass. CPU-only peakRSS≤2,076,288KiB; noGPU/PTX/network/agents/update/publication. Full report/audits/logs: `/workspace/exports/pocket-shared-attention-20261003/report.md`.

## Transformer input-adjoint follow-up

Test-only serial linear dInput reductions preserve CPU dWeight/dBias and leave1,206originalPviolations. Combining prior forward-linear/GELU with serial dInput leaves940; adding hostF32GELUderivative leaves939. Local input-fixed adjoint differences≤1.1175870895385742e-8have no opposite nonzero signs; propagated selected parameter errors still exceed4e-6. ControlP/M/Vreproduce savedCPU. Best original-outlier count remains prior934; new failures elsewhere are unscanned.

A distinct forwardLayerNorm rounding intervention usesF32sqrt thenF32reciprocal, whereas CPU computes reciprocal/sqrtF64then rounds once. Added to prior forward-linear/GELU it leaves1,041originalPviolations. Earliernormadjointcheck usedfixedCPUinvanddidnotcoverthisboundary. CPUbackwardremainsunchanged; these mixed hostformulas cannotqualifyactualdevicesqrt/exp orfullnativeAD. Productionarithmetic/tolerancesunchanged.

Newtiny independentF64linear/norm/GELUformula andactualCPUtrunkchecks passrepeat/race; affectedvet/hostCLIbuild/format/diffpass. CPU-onlydiagnosticpeakRSS≤2,071,872KiB; noGPU/PTX/network/agents/update/publication. Full report/audits/logs: `/workspace/exports/pocket-shared-adjoint-20261003/report.md`.

## Flow-forward and transformer-seed follow-up

A paired explicit CPU flow backend control isolates local-gradient/outer-objective scaling from default scratch seed scaling. It leaves1,209originalPviolations andmatchespriorlocalaccumulationnorm; its freshlycomputedPdiffersfromdefault at1,251selectedpoints. StoredcontrolP/M/VreproduceoriginalCPU. Test-onlyserialflowforwardprimal/tangent linears leave837violations/58oppositesigns, maximumPerror7.5350e-5. Combining priortransformerforwardlinear/GELU leaves628/51,max9.4748e-5. Flowforwardnormsqrt/reciprocalrounding leaves851; newoutlierselsewhereunscanned.

Eachnewmodeexecutes78flowcalls(52ordinary/26dual),CPUordinary/dualbackwardretained, endpointsdiscardedbyexistingLSDorchestration. No expectedvaluesenterAD. Tinyordinary/timeindices0/1/result/namedgradients andindependentF64population/samplevarianceprimal/tangentchecks passrepeat/race; modelvet/hostCLIbuild/format/diffpass. Hostexp/sqrt andactualnativeseedboundariesremainunverified; productionarithmetic/tolerancesunchanged. CPU-onlypeakRSS≤2,579,776KiB; noGPU/PTX/network/agents/update/publication. Fullreport/audits/logs: `/workspace/exports/pocket-shared-flow-20261003/report.md`.

## Final condition and dual norm adjoints

The test-only final flow condition-projection serial reduction observes 79,872 values across 78 calls, with CPU/serial difference at most 5.587935447692871e-9 and no local opposite signs. Its propagated normal/combined variants leave 1,205/636 original P violations, compared with paired controls 1,209/628. The combined maximum error remains 9.185867384076118e-5. Expected values stay isolated from AD.

Source-formula dual norm adjoints at CPU forward statistics have zero differences in all 119,808 value and 119,808 tangent observations across 234 calls. Norm-weight multiplication order changes values by at most 2.9802322387695312e-8, but no selected predicted P changes. Normal/combined counts stay 1,209/628. Ordinary backward and other CPU adjoints remain intact; actual device statistics/transcendentals are unqualified.

New tiny independent F64/dyadic/rounding-bound/norm-JVP finite-difference and complete flow-boundary checks pass repeat/race; model vet, host CLI build and formatting/diff pass. CPU-only peak RSS is at most 2,580,480 KiB. No GPU/PTX/network/agents, optimizer update or publication occurred. Full report/audits/logs: `/workspace/exports/pocket-shared-condition-20261003/report.md`.

## Complete flow adjoints and full parameter distribution

Test-only serial flow linear-backward/SiLU-backward formulas leave 639/640 original violations when added separately to prior combined forward. Joint formulas leave 617 original failures and 49 opposite signs. Input-fixed linear adjoints differ at most 1.4901161193847656e-8 and SiLU value adjoints at most 3.725290298461914e-9, with no local opposite signs; sampled weight contributions and dual SiLU tangent adjoints match. Production arithmetic is unchanged.

A new streaming scan compares all 89,449,730 diagnostic predicted parameters with pinned saved CPU/native checkpoints. It reproduces the original 1,293 failures and finds 790 joint-variant failures against Vulkan: 617 original plus 173 new. Maximum error is 9.043398313224316e-5; full RMS is 3.178265633951429e-8. The joint variant also differs from saved CPU at 483 values above 4e-6. Selected-outlier improvement did not establish acceptance. Reference values never enter AD or model updates; complete native moments/audio remain untested.

New tiny independent F64 flow adjoint/boundary/policy and streaming-reader rejection checks pass repeat/race; model vet, host CLI build and formatting/diff pass. CPU-only peak RSS is at most 2,611,200 KiB. No GPU/PTX/network/agents, optimizer update or publication occurred. Full report/audits/logs: `/workspace/exports/pocket-shared-flow-backward-20261003/report.md`.

## Time inputs and composed seed routing

New input-only pinned replay records 9,984 released time/frequency/branch entries. Maximum angle is 0.9276309609413147 radians; host sin/cos/JVP meets unchanged reference gates. All 59,392 released-geometry RoPE outputs match the scalar convention. Actual device sin/cos/exp/sqrt results are unavailable; input/source agreement cannot qualify those instructions.

Metadata-only composed backbone audit passes for 807 commands, 65 named parameter bindings and 79,690,978 values, checking condition seed padding, single routing, complete arithmetic/ID remapping and final target join. Terminal gradient-view reshape order can vary from map iteration; those exact views are matched by name without permitting arithmetic reordering. Deliberate routing/binding/scale corruption rejects in tiny repeat/race tests. Model vet, host CLI build and formatting/diff pass. No native/GPU/PTX execution or production arithmetic change occurred. Evidence: `/workspace/exports/pocket-shared-time-boundary-20261003/report.md`.

## Offline native-overhead and export accounting

The later graph inventory contains 661 resident allocations plus 215 owned tape arenas: 876 buffer owners, 1,142 tensor views, 395 dispatch stages/descriptor sets, 1,841 descriptor bindings and seven plans/command buffers/fences/descriptor pools. Persistent kernel objects are additional. The retained Go tape-object size floor is 226,644 bytes before maps, allocator classes, builders and runtime overhead.

Tape-only resident/later payload remains 2,140,716,680 bytes, leaving 6,766,968 below 2 GiB. Hypothetical per-buffer 4 KiB rounding adds 831,864 bytes and fits; 64 KiB rounding adds 33,112,440 and exceeds the cap. Neither gives an actual `vkGetBufferMemoryRequirements.size` bound. Opaque driver pipeline/descriptor/command/fence memory and native phase RSS are unmeasured.

Source lifetimes close row children and their transformer/EOS producer before the later graph, and close successful later tapes before update/checkpoint/export. Shared checkpoint writing borrows the private host boundary. The CLI defers resident/kernel cleanup until after export; terminal failures retain uncertain owners. The CLI sets separate resident and tape ceilings and does not call `VulkanSetMemoryBudget`, so it installs no combined 2 GiB native allocation guard. Explicit native allocation charging uses actual requirements when an operator budget exists; queried heap/allocation limits always apply. Conservative host mapping admission remains `ResidentUpperBytes + 9*ParameterBytes + 2*deviceCap` and is unchanged.

A new CPU-only owned-restore/export-selection ledger adds 80,217,348 bytes for frozen Mimi decode and zero duplicate F32 payload for Flow EMA selection (127 Flow + 87 Mimi tensors). It forecasts 5,381,339,428 bytes including separately forecast native resident and mapping payload; pre-GC HeapAlloc plus those forecasts is 5,382,750,532. Actual CPU-only peak RSS is 2,934,576 KiB. No export writer/publication or native allocation executes; header/writer, driver overhead and actual native coexistence RSS are unqualified. Attribution-only GC occurs after the pre-GC sample and does not change production policy.

Saved-log reconciliation, without rerunning restore, gives completed-later forecast 5,620,689,504 bytes and owned-restored-workspace/later forecast 5,619,093,072 after subtracting the tape-only branch. The first scenario script overstated the latter by 4,624 bytes; original output is preserved beside corrected `resource-reconciled.json`. The restored forecast excludes actual later-source Values and full row-child coexistence. These figures cannot lower admission.

Metadata inspection passes repeat10/race; released slot inventory and export selection pass once. After renewal, affected Vulkan/model/CLI vet, host CLI build, resource-test formatting, staged/unstaged diff checks and saved-log reconciliation pass. The build binary was removed without execution; staged SIMD remains eight files/469 insertions. No completed numerical/inference/restore runs were repeated. Report/audits/raw evidence: `/workspace/exports/pocket-shared-resource-overhead-20261003/report.md`. No GPU/PTX/network/contact/delegation, optimiser update, fallback, terminal retry or publication occurred.

## Approved bounded native capture

Rui approved a bounded Vulkan resource/intermediate protocol. A new opt-in test queries all 876 saved buffer extents before allocating storage, applies a process-local joint 2 GiB/876-allocation budget, page-touches the payload and constructs seven descriptor-interface plans without submitting them. On pinned RTX3060/driver610.57.04, actual requirements total 2,140,717,168 bytes: 488 bytes padding and 6,766,480 bytes margin. All allocations use HOST_VISIBLE|HOST_COHERENT flags on heap1. Maximum sampled resource-only RSS is 2,338,164,736 bytes; final explicit native bytes/count are zero. Kernel/interface-plan RSS differences include Go/driver retention. Live training arrays, row-child coexistence, checkpoint/export I/O and actual CLI phase RSS are excluded; production admission and budget wiring are unchanged.

A separate one-shot existing time-shader capture uses the 9,984 saved input rows in 78 blocks, with 390 buffers/280,800 native bytes and three plans. It fails the unchanged value gate: 6,390 of 19,968 cosine/sine values exceed 6e-8, maximum 3.129243850708008e-7, RMS 8.71617463188689e-8. All 19,968 tangents pass2e-5; maximum 2.384185791015625e-7. Cosine contributes447 failures, sine5,943. Independent F64 trigonometry at exact saved F32 angles confirms value drift; no expected value enters device arithmetic. All explicit owners close before the numerical failure is reported; no retry follows. Fullstep P gate4e-6 and the saved1,293 failures are unchanged.

New offline probe validation passesrepeat10/race, opt-in skip guards/vet/compile pass; staged SIMD stays8files/469insertions. Pinned health checks before/after report idle GPU and no compute processes, uncertainty or device loss. No full training/AD/optimiser update/PTX/fallback/publication executes. Failed test-only attention field compilation and initial JavaScript F32-decimal analysis errors are preserved; corrected saved-result analysis restores F32 with Math.fround and reproduces the6,390 count without native rerun. Report/raw evidence: `/workspace/exports/pocket-native-capture-20261003/report.md`.

## First-row composed flow capture

After renewal, a one-shot no-update capture evaluates diagonal, primary JVP and endpoint forward graphs at shared CPU trunk/latent boundaries, each with device-time and explicit native host-time control. Complete sources119,008,416/119,004,280B are admitted before forward-only diagnostic roots are pruned; actual native payload39,473,568/39,470,464B fits the isolated2GiBbudget. Sixgraphscapture85/83value-JVPpairs each; no reverse/loss/accumulation/optimiser executes. CPUprimarydefinesendpointinput,sofullnativeLSDchain/trunkparityareexcluded.

Paired input/condition projections are exactly equal. First downstream value separation is time0FC1; primary tangent separation is time1FC1. Device-time final output maxerrors are1.049041748046875e-5(diagonal/endpoint) and5.245208740234375e-6(primary); corresponding native host-time controls are≤3.337860107421875e-6. Each device-time branch hasoneof32outputsabove4e-6diagnostically;controlsnone. PrimaryJVPdevice/controlmax3.516674041748047e-6/9.5367431640625e-7; pairednativeJVPmax4.172325134277344e-6. Intermediate pairedvalueerrorreaches6.67572021484375e-5 atendpointfinalmodulation. Time-roundingnowhasmeasuredforward/JVPpropagation;parameter/backwardeffectsremainunmeasured. Host-timecontrolisdiagnosticonly,notproductionfallback/acceptance.

Capture2.15s/peak674,052KiBRSS; eachgraphclosesbeforethenext/finalexplicitbytes/count0, no uncertainty/device loss/retry. Newtinygraphadmissionandevidencehelperspassrepeat10/race;skipguards/vet/compile/format/diffpass. InitialCPUpreparationcwdfixturefailureispreservedandcorrectedbefore nativework. StagedSIMD8files469insertions,caps/admission/tolerancesunchanged. Report/rawtraces: `/workspace/exports/pocket-flow-native-boundary-20261003/report.md`.

## Reverse/condition-seed preparation and occupied GPU

The renewed no-update reverse protocol reuses pinned CPUcondition/nativeforwardoutputs, loads onlyreleasedflowF32 andderiveslogvar0lossseeds forfirstrowdiagonal/endpoint/primary. Complete source119MBpassesbeforeendpointparameteroutputpruning; diagonal/primarynativepayload79.276MB/477stagesdevice-time and79.272MB/471stagescontrol; endpoint40.126/40.122MBwith407/401stages.66namedparameterswillcompareeveryvalueagainstindependentCPUroutines atidenticalactualseeds. Endpointdirectparameter/condition/timegradientsstopped;capturedendpointDInputwouldfeedprimary. No nativecapturehasruninthisslice.

Newtinyseed/interpreterCPUgradient/endpointstop/manifesttestspassrepeat10/race;additionaloverflow/stoppedDInput/F32bytestestspassrepeat10/race;guards/modelvet/compilepass. CPUadmission0.35s/77,952KiBpeakRSS. Initialinputboundfailurepreserved:priorflowtrace13,481,246Bexceeds8MiB;extractpinned9,522Bfinal-output-onlyfixtureunder64KiBboundinstead ofincreasingcap. StagedSIMD8files469insertions/caps/arithmetic/admission/tolerancesunchanged. Report/protocol: `/workspace/exports/pocket-flow-native-reverse-20261003/report.md`.

NativeexecutionheldbecausefullGPUprocessreviewfindsWPEWebProcessgraphicsworkload; latest55%utilisation/131MiBbrowser/183MiBtotal despiteemptycompute-onlyquery. No kill/reset/recovery/contact/nativeinitialisationoccurred. Re-auditallgraphics/computeprocessesonceGPUidlebeforetheone-shotreversecapture;do notrepeatcompletedofflinepreparation.

## Resumed first-row reverse capture — 4 October

Rui requestedresumeGPUwork;fullgraphics/computehealthnowidle onpinnedRTX3060/610.57.04 andpriorWPEworkloadendedwithoutintervention. Runpreparedone-shotreverseprotocolonly, no completedoffline/forward/fulltraining reruns. Sixgraphsclosecleanly,finalexplicitbytes/count0,noInFlight/Uncertain/DeviceLost. Capture3.08s/peak577,656KiBRSS,maxnative79,276,640B, belowunchanged2GiB/6GiBcaps. Productionarithmetic/admission/stagedSIMDunchanged;nooptimizer/PTX/fallback/publication.

All66namedflowgradients(9,758,752values)compareagainstCPUatidenticalper-variantactualseeds. Diagonaldevice/controlmax1.5974044799804688e-5/5.7220458984375e-6,RMS8.8809e-8/3.0583e-8,diagnosticabove4e-6counts628/1. Primarydevice/controlmax1.8477439880371094e-6/1.3541430234909058e-6,zeroabove4e-6,13/8oppositesigns. Pairednativegradientmaxdiagonal9.5367431640625e-6/primary3.135204315185547e-5;pairedchainsuse differentlossseeds andinclude both seed/arithmeticdrift. Endpointdirectparams/condition/timesstopped,onlyDInputfeedsprimary. Firstrow(diagonal+primary)conditioncontribution/13pairedmax1.5468685887753963e-8 over1024values;actualallrowseeds/trunkbackwardunmeasured. Diagnosticcountsdo notacceptfullstep4e-6orhosttimefallback.

Savedanalysisvalidatesfullnamedmanifests/F32extents/finitevalues/closure;posthealthidle/no processes. Four39,035,008BgradientfilesandJSON/audits: `/workspace/exports/pocket-flow-native-reverse-20261004/report.md`. No clipping/norm/EMA/update/modelwriter/CLI/nativefailure/retry occurred.

## First-row objective-seeded native transformer capture

New29rowstandalone transformerforward/VJPusesCPUconditionersequenceandcapturedfirstrowseedswithcorrect.75/13diagonal+.25/13primaryweights;prefix15/allotherrowseeds/EOSzero. Earlier(diagonal+primary)/13trace was diagnostic-only. Complete source1,022,105,832B/physical672,110,824B admittedbefore native;actual672,111,136B/293stages/50params75,524,096gradientvalues each. CPUconditioner excludesnativeprojection;first-row/allrow limitations explicit.

Allnativegradientssame-seedCPUmax5.122274160385132e-8,RMS4.7801e-10/4.8453e-10,zero diagnosticabove4e-6,55/56oppositesigns. Nativefirstconditionmax3.814697265625e-6 versuspinnedCPUcondition;allforwardoutputmax6.67572021484375e-6(twoof29,696above4e-6diagnostically). Pairedfirstrowseedmax6.05359673500061e-9 propagatesintopairedgradientmax9.371433407068253e-8/66oppositesigns;inputVJPpairmax1.1548399925231934e-7. Identicalnativeforwardoutputsacrossseedvariants;fullnamedscanvalidatedF32files/offsets/extents/nonfinites.

Capture7.58s/peak2,165,412KiBRSS;allnativeallocationsclosebetweenvariants/finalbytes/count0/nouncertainty/loss/retry. FirsthealthWPEbusyrejectedbeforeruntime;afterofflineanalysertestsonefreshcheckidleandpostcaptureidle. Tinyassembly/seedpad/objective/interpreterCPUgradienttestspassrepeat10/race/guards/vet/compile;streaminganalyserselftestpasses. Caps/arithmetic/admission/stagedSIMDunchanged,nooptimiser/PTX/fallback/publication. Report: `/workspace/exports/pocket-transformer-native-20261004/report.md`;two302,096,384Brawgradientfilesretainedworkspace,compactbundlecontainsmanifests/statisticsonly.

## Native conditioner and all forward conditions

A newboundedno-updatecomposedbackboneforwardadmitscomplete1,075,125,156Bsourcebeforesequence/outputpruning,actualnative346,851,616B/408padding. Nativevoice/text/BOS/shiftedaudioprojections/assemblyandsixtransformerlayerssupplyall14conditionrows. SequenceCPUerror≤5.960464477539063e-8,alloutput≤6.67572021484375e-6,14,336conditionvalues≤4.172325134277344e-6(one diagnosticabove4e-6,row2);firstrowmax3.5762786865234375e-6. CPUreferencecomparisononly,fullstepgateunaccepted. CapturednativeconditionJSONpin633428297efdf349bd136e17262d0ee253999269d6fb3f65f78f6e805537f722.

Capture3.32s/peak968,508KiBRSS/finalnativebytes/count0, no uncertainty/loss/retry. Newtinygraph/CPUinterpreter/source/token testsrepeat10/race/guard/vet/compilepasses,fullsourceadmission/savedF32countanalysispasses. Prehealthidle;posthealthtwoWPEgraphicsprocesses31/67MiB,relativestarttimeunknown:timingnotcontrolledefficiency evidence, furthernativeworkrequiresfreshidleaudit. No reverse/optimizer/PTX/fallback/publication/cap/tolerancechange,stagedSIMDunchanged. Report: `/workspace/exports/pocket-native-conditioner-20261004/report.md`.

## Native all-row flow and joined condition seeds

A newno-updatereplayusesall13actualnativeconditionrows,exactpinnedsamplingandnativeprimary→endpointlatentchain. FivefusedproductionboundarygraphsperrowplusEOS/reduction/padtotal68executions;fullsources119–238MBadmittedbeforeparameteroutputpruning. Diagonal.75/13alreadyapplied/distill.25/13appliedonce/EOS.1singleVJP;initialMLPlogvars0verified. Endpointdirectgradientsstopped. NativeconditionreductionCPUmax5.448237061500549e-8/RMS3.78499672907719e-9,targetmax1.9278377294540405e-7,EOSconditionmax2.3283064365386963e-9;all0diagnosticabove4e-6/oppositesigns. Joinedflow+EOSseedmax5.448237061500549e-8vsCPUatidenticalnativeconditions. CPUreferencearrayscomparison-only,no parameteraccumulation/update/fullstepacceptance.

Capture23.78s/peak1,149,416KiBRSS,maxnative40,127,760B,eachgraph/finalledger0/no uncertainty/loss/retry. Prefix/flowpadding/singlejoinroutingverified;firstinvalidEOSseedallowed. NewfusedchainandEOSweight/paddingtestsrepeat10/race/guard/vet/compile/source/savedanalysispass;unusedimportcompilefailurepreservedbefore runtime. InitialhealthWPEbusyheld;freshcheckidlebeforeone-shot/postidle. Caps/arithmetic/admission/stagedSIMDunchanged/PTX/fallback/publicationexcluded. Savedallrowseedpin aa7e79fdf84b727936a462e7d69cbdb31a6bf153d0183f8adbb2eccd2251d6b6,report `/workspace/exports/pocket-native-allrow-20261004/report.md`.

## Complete backbone VJP at all-row seeds

Two complete nativeconditioner/transformerVJPgraphsusecapturedallrownativeflow+EOSseedandlabelledCPUrow-reference-seedatsharednativeconditions. Source1,075,125,156B/actualnative706,566,288B/492padding/55params79,687,712values foreachvariant. CPUFlowLMsame-seed reference zeroEOSlogitseedavoidsdoublecounting;EOShead/flow/weightMLPparams excluded. Once-native-directtargetjoin followsnormalizedVJP.

AllgradientCPUmaxnative-seed6.258487701416016e-7/reference-seed5.066394805908203e-7,RMS2.0869e-9/1.9722e-9,0diagnosticabove4e-6,52/46oppositesigns. Pairedgradmax2.294778823852539e-6/RMS7.7612e-9/141oppositesigns,largestinput_linear.weight;nativeforwardoutputsidentical. Voice/normalized/targetjoinedCPUerror≤5.21540641784668e-8. Fullstreamed55namemanifests/F32extents/finitecountsvalidated;no globalclip/update/Pacceptance.

Capture7.26s/peak2,154,612KiBRSS/allclosure0/no uncertainty/loss/retry. Newtinyrepeat/race/guards/vet/compile/sourceadmission/savedscanpasses,caps/arithmetic/admission/stagedSIMDunchanged. Prehealthidle/postWPEgraphics72MiB,starttimerelativecaptureunknown,noefficiencyclaim/furthernativefreshhealthrequired. Rawtwo318,750,848BF32filesretainedworkspace,compactbundlemanifests/statistics. Report `/workspace/exports/pocket-native-joined-backbone-20261004/report.md`.

## Complete corrected gradient inventory and no-mutation prediction

Newallrowfullgradientcaptureincludes66flow/8weight/2EOSparams atsavednativeconditions/forwardboundaries. Productiondiagscaleis`.75 * F32(1/13)`=0.05769231170415878; earlierseed-onlyprobeused`F32(.75/13)`=0.057692307978868484. Preservepriorapproximate-scale evidence; correctedjoinedseedchanges12,431/29,696values(max3.026798367500305e-9),newcorrected-seedbackbone55params completes131names89,449,730values. No productionarith/tolerancechange. FlowCPUmax5.282461643218994e-6/9diagnosticabove4e-6/31oppositesigns;weightexact/EOSmax1.862645149230957e-9;corrected-backboneCPUmax7.003545761108398e-7/0above. FullunionRMS5.290258477206675e-9/93oppositesigns. Referencesuseactualsharednativeboundaryinputs,notoriginalallCPUcomposedoracle.

F64compensatednative/referencegradientnorms14.756524060160528/14.756525947107404. Nativeclip-onlyprobeendsafterallocation/beforefirstsubmissionwith`resident norm extent`:testcallerreversedNormStatsdestination/source. No native retry/recovery/update follows; futurecallcorrected/newmockwiringrepeat/racepasses. Earlierlocale-sortedinventorypreinitrejectioncorrectedtoGobyteorder,allfailurespreserved. Nativeclippingunqualified.

Independentread-onlyfirststepAdamWpredictionusingF64-clippingscale scansall89,449,730parameters:vs savednative0above4e-6/max2.9802322387695312e-8;vs savedCPU1,293above/max1.0317843407392502e-4. Same-boundaryCPUrefgradientpredictionvs savedCPU1,094above/max1.233760267496109e-4. Original1,293reproduced; no per-locationoriginal/newoverlaprecorded. Capturedgradientsreproduce savednativePwithin gate,butnativePacceptanceagainstCPUstillfails. No optimisercall/modelmutation/EMA/writer/CLI/PTX/fallback/publication. Missingparametercapture17.14s/peak1,235,436KiB/correctedbackbone5.18s/peak1,821,992KiB;successfulownersclosure0. CPU-onlyprediction8.27s/peak625,024KiB. Newtinyrepeat/race/guards/vet/compile/fullinventory/format/diff/indexpreservationpass,staged8files469insertionsunchanged. Report `/workspace/exports/pocket-native-parameters-20261004/report.md`;rawF32retainedworkspace,compactbundlemanifests/statisticsonly.

## Renewed native clipping and narrow-domain candidate

Rui renewed work after the norm-call failure. One fresh corrected NormStats/maximum/fixed-maxsum/clip capture passes2.25s, native decodednorm14.756524147708895 vsF64reference14.756524060160528(error8.75483667783783e-8). Native/F64/priorPpredictionclipscale haveidenticalF32bits1032505660(value0.06776663661003113); priorfullPscanreusedwithoutrepeat. Allnativeownersclose/finalledger0,17MiBallocation/256MiBphaseRSScaps unchanged;sampledRSS236,343,296B/wrapperpeak246,828KiB. Nativeclippingboundarymeasuredforcompletecapturedinventory;1293CPUreferencePviolationsremain. Report `/workspace/exports/pocket-native-norm-renewed-20261004/report.md`.

A separateexternaltest-onlyFMA-HornerTaylorcandidate(sindegree13/cosdegree14,domain|angle|<=1) passes16385signedCPUanglesrepeat10/race and9984pinnedinputchecks, unchangedlocal6e-8value/2e-5JVPgates. SPIRV1.3/Vulkan1.1interface/validationpasses/noSinCosinstructions;productionGLSL/SPV/routesunchanged. Candidate-onlynativecapture19968valueshas0above6e-8,max5.960464477539063e-8/RMS4.746221504752494e-9;JVP0above2e-5/max5.960464477539063e-8. IndependentF64trigmax5.388438584219557e-8. Previousbuiltin6390valueviolationsfromsavedbaselineonly,notrerun. Candidate0.37s/peak193,672KiB/actualnative280,800B/allclosure0/no failure/retry. Fullgraphicshealthidlebefore/after. Broadproduction8192anglecontractnotcovered,cannotpromote. No model/update/PTX/fallback/publication;caps/tolerances/admission/stagedSIMDunchanged. Report `/workspace/exports/pocket-time-polynomial-20261004/report.md`.

## Production arithmetic corrections

Rui requested implementation through the failing arithmetic path. Production time GLSL/SPIR-V now uses explicit FMA split-pi/2 reduction and polynomials across the unchanged `|angle| <= 8192` admission. The original experimental call-site replacement had missed both main paths; corrected executable tests reject surviving built-in Sin/Cos. Host 555,588 signed angles and native 32,768 full-range values pass the `6e-8` value gate. Released 19,968 values and JVPs have zero violations at unchanged `6e-8`/`2e-5`. The composed read-only parameter comparison reduces the original 1,293 failures to 418.

Production linear forward/JVP now uses independent FMA lanes and a balanced reduction. Independent CPU SIMD tests match within the existing fixture gate (measured maximum zero); F64 error/sumabs is 9.94e-8. Conditioner maximum error falls to 1.43e-6. Regenerated all-row, flow/weight/EOS, backbone and native clipping captures close successfully. Across all 89,449,730 gradients, matched-boundary CPU comparisons have zero values above 4e-6; read-only first-step prediction against the original independent CPU checkpoint still has **45 parameter failures**, maximum **8.922070264816284e-6**. Report: `/workspace/exports/pocket-linear-correction-20261004/result.md`.

An additional compensated-normalisation/input-adjoint experiment then timed out with `Vulkan submission still in flight` during composed conditioner execution. No native retry or recovery followed. The unqualified changes were removed from production and retained in the evidence directory. Native execution is stopped at this safety boundary. The last completed production composition is the 45-failure result.

Complete affected portable Vulkan, Pocket TTS and pockettrain suites, scoped repeat10/race and vet pass after restoration. A stale working-set audit omitted the released pad-token embedding row; its fixture now includes the row and the correct 89,449,730-element inventory. Numerical gates, caps, admission and staged SIMD are unchanged.

## Next work

Obtain actual native requirements/opaque overhead and complete producer/row-child, restored/completed/checkpoint/export phase RSS evidence before revising admission; offline payload and lifetime accounting now cover the later graph and export selection. Trace the remaining 45 parameter failures using saved boundaries and offline reference checks; resolve the incomplete-submission safety boundary before further native work. Preserve the completed scoped ownership/error coverage and audit native execution only after resource and numerical blockers are resolved. Full released Vulkan/PTX numerical acceptance, short/long efficiency profiles, cancellation/soak/resume and Samantha adaptation remain open. Publication/listening holds and the no-contact/no-delegation instruction remain.
