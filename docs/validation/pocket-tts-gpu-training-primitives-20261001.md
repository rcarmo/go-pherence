# Pocket TTS training implementation — 1 October 2026

Added Vulkan/PTX AdamW/EMA, linear-backward and SiLU-backward primitives, plus native import and whole-utterance preparation for the private Spock/Samantha datasets. Complete device loss/backward orchestration is still unfinished; the training CLI exposes separately named experimental attention-backward, flow and transformer hybrids and rejects full-device GPU requests.

## GPU primitives

- `backends/training/adamw.go`: shared finite-value/shape admission and host-computed F32 bias corrections matching the CPU trainer's float64 power calculation. Global clipping scale is supplied explicitly; kernels do not substitute per-tensor clipping.
- `backends/vulkan/vulkan_training.go` and `shaders/training_*.glsl/.spv`: experimental resident-tensor stages for parameter/moment/EMA updates and overwrite dX/dW/dBias. Stages fit the existing `VkF32Plan` submission/lifetime contract.
- `backends/nvidia/ptx/training.go` and `runtime/training.go`: independently owned PTX module, checked pointer extents/aliases, synchronous update/gradient wrappers. An attempted AdamW launch/synchronisation failure poisons the update operator; no retry is admitted.
- `vulkan_spirv_contract.go`: admit only the additional GLSL Sqrt instruction needed by AdamW. Unrelated unary extended instructions are still rejected.

All mutable optimiser tensors are disjoint. Linear outputs may not overlap inputs; read-only inputs may overlap each other. Initial linear bounds are M<=1024 and K/N<=16384. Each output has one owner, with no float atomics. Branch-gradient accumulation is caller-managed. Kernels require finite input state, nonnegative second moments and representable arithmetic; device buffers are not host-scanned by these low-level wrappers.

GLSL uses precise arithmetic and PTX separate round-to-nearest instructions. Reduction order differs from larger SIMD/tiled matrix operations. A host schedule model was checked against the pinned independent PyTorch fixture `model/pockettts/testdata/training_step_pytorch.json`, upstream `0acce6b2f390150267557770d2098c5caa9a18ac`. Across 1,212 optimiser/EMA values, maximum absolute error was 1.1920928955078125e-7 and RMS 6.262139032995525e-9, below the unchanged existing CPU optimiser tolerance of 4e-6. This executes host arithmetic only. No device output error distribution or performance measurement exists yet.

The AdamW and linear-backward GLSL shaders compile with `glslangValidator -V -S comp` and pass `spirv-val`; both PTX entries assemble using CUDA 12.8 `ptxas -arch=sm_80`. Offline compiler execution does not initialise CUDA or Vulkan. Ten focused tests passed ten times (100 top-level passes) and under race checks, with focused vet. Final gate: **19:59:52.329553715–20:00:04.135210660 UTC**, exit 0. Foreign Linux ARM64/RISC-V package builds passed; foreign execution was not attempted.

SiLU backward was added next for both backends, with a stable sigmoid branch to avoid finite-negative-input exponential overflow. Its GLSL compiles and validates, and PTX assembles. A host derivative-expression check against independent F64 central differences, Vulkan metadata/alias checks, PTX fake launch ABI tests and expanded shader reflection tests passed ten times (40 top-level passes), plus race and vet. Gate: **20:12:31.048273515–20:12:43.728560210 UTC**, exit 0. PTX uses `ex2.approx`; Vulkan uses the device Exp instruction. Actual device approximation error remains unmeasured.

Evidence: `/workspace/exports/pocket-gpu-training-primitives-20261001/`. Initial missing fake Vulkan push-function setup and fixture buffer/trainable-name mismatches were corrected; failure logs are preserved. These were test setup defects, not device runs.

## Whole-utterance datasets

Private Gitea source revisions inspected: Spock `4d0d0299cce90ad45a9ce74eefba43d2ab7b74b4`, Samantha `54bf3770250f6601ad5087a3c184ddb818517bb3`. No source audio was downloaded or altered.

`cmd/audio/pockettrain/import_whole.go` adds `-mode=import-whole`. It verifies source and exported split SHA256SUMS, declared checkpoint/manifest identity, exact source transcript files, waveform pins, exclusions and train/dev ID/audio leakage. Output contains only portable root-relative training rows. Outputs inside the immutable source checkpoint, including symlink redirection, are rejected. The importer does not run the repository exporters or issue a quality QA attestation.

`whole.go` and prepare integration add `-manifest-format=whole-utterance` with a required derived manifest digest. Complete targets use the shortest distinct complete training-only reference that fits 62 frames. At least two eligible references are required. No word timestamps, implicit cropping, dev conditioning or audio preprocessing is added. Target ceiling: 374 valid frames (29.92 seconds), plus one zero EOS row. Four Samantha clips exceed that envelope; reviewed splits or exclusions are necessary before their import.

The provenance policy is `whole-utterance-shortest-distinct-train-reference-ceil1920-normalized-target-eos-pad1-v1`, distinct from aligned first-cut preparation. Final listening/transcript/scene QA and consent/rights handling are separate from engineering admission. The source repositories are private research excerpts with no redistribution licence asserted; keep datasets and synthetic voice outputs private.

Seven focused whole-utterance/import tests passed ten times (70 top-level passes), plus race, vet and host build. Final gate: **20:08:27.362086971–20:08:29.912270665 UTC**, exit 0. Tests use repository-generated tiny canonical WAVs and a supplied fake codec; no real model/recording was loaded. Earlier whole preparation gate included Linux ARM64/RISC-V builds and source-layout tests (5 Bun tests, 3,008 assertions).

Evidence: `/workspace/exports/pocket-whole-utterance-20261001/`. Usage is in `cmd/audio/pockettrain/README.md`.

## Normalisation, attention and mixed derivatives

Added LayerNorm backward for dX/dWeight/dBias and causal attention backward for dense dProbability, scaled dScore, dQ, dK and dV on both backends. Layouts are `[S,H,D]` for Q/K/V/dOutput and `[H,S,S]` for probabilities and scratch. All masked derivatives are explicitly zeroed. Fixed-key reductions admit only queries at or after that key and within the requested context. dScore includes the score scale once; inverse RoPE follows dQ/dK accumulation separately.

LayerNorm consumes pre-affine normalised values and cached inverse standard deviations, so zero affine weights do not make the derivative ambiguous. Its initial dX kernel assigns a complete row to one invocation and uses serial reductions. This avoids atomics and barriers but is an unoptimised baseline.

Vulkan composes five attention stages or three norm stages into one validated `VkF32Plan`. PTX provides a complete five-stage attention pass with whole-pass extent/alias admission before any launch; each stage currently synchronises. Stage failures stop immediately. Neither path has a speed or residency qualification.

Ten new backend tests passed ten times (100 top-level passes), race, vet and Linux amd64/arm64/riscv64 builds in **20:25:49.090899102–20:26:02.298606620 UTC**, exit 0. Host attention schedules matched independent F64 finite differences across 516 derivative values: maximum absolute error 3.70545433781011e-8, RMS 4.2827431726522586e-9. LayerNorm's 25 values had maximum error 4.507420478527813e-8, RMS 1.628877429377043e-8. Tests cover context 0/1/2/full and deliberately nonzero masked probability cells. These derivative-expression checks do not execute device kernels or establish device tolerances.

A read-only second-model review confirmed the formulas, mask/index constraints and pre-affine LayerNorm requirement. A later full adapter/ownership review timed out after 60 seconds; it returned no findings or approval.

### Model integration

`model/pockettts/training_attention_backend.go` defines an explicit attention-only backward interface. CPU forward and all other derivatives remain on CPU. `PTXTrainingAttention` and `VulkanTrainingAttention` implement transfer-inclusive calls with byte ceilings and owned outputs; they require caller-owned live kernel operators and never initialise a device themselves. Native operator readiness, cancellation, shape and budget failures are rejected before allocations.

Transformer and FlowLM backward now propagate backend errors with no CPU fallback. Returned derivatives must be finite, correctly sized, mutually disjoint and disjoint from inputs. The default nil backend preserves CPU execution. The CPU replay runner rejects hybrid injection at construction and before updates, preventing hybrid arithmetic being saved under a SIMD-only identity.

Four model-interface/replay tests passed ten times (40 top-level passes) plus race in **20:30:55.274178799–20:31:05.067511273 UTC**, exit 0. The changed CPU path also passed its existing transformer/FlowLM finite-difference and full-step/PyTorch AdamW-resume tests with unchanged tolerances. Host readiness rejection then passed ten times and under race after its addition. Transfer-success and native allocation/cleanup behaviour have not been exercised; no device calls were authorised.

Adapters retain buffers/arenas/plans after uncertain launch/transfer failure rather than freeing storage potentially referenced by unfinished work. They become terminal and expose no automatic reset/retry. PTX failed frees retain ownership too. These conservative paths may retain allocations until process-level disposition; they are not leak-free recovery or a crash-durability proof. Per-call allocation and transfer are deliberate baseline costs; a persistent workspace remains to be implemented and measured.

### Minimal-LSD time-JVP

Added analytic dual SiLU backward to both backends, including the second derivative for reverse-over-forward differentiation. It produces both primal and tangent input adjoints from independent output seeds. It uses no finite differences during training. The host expression check uses independent finite differences of the forward scalar objective: 28 values, maximum absolute error 1.9495495007237906e-7 and RMS 4.9296605406616424e-8. Three new tests passed ten times plus race in **20:33:24.402251065–20:33:37.542767583 UTC**, exit 0. GLSL/SPIR-V validates and PTX assembles, including the combined six-entry runtime module.

New evidence and source snapshots: `/workspace/exports/pocket-gpu-training-backward-20261001/`. The initial attention shader used reserved GLSL identifier `active` and compiler output outside the existing closed SPIR-V opcode envelope. Renaming it and expressing inclusive bounds as `<query+1` fixed compilation/admission without widening that envelope. Prior logs and gates are retained. Source layout checks passed (5 Bun tests, 3,008 assertions), as did affected vet and host/foreign package builds.

## Dual flow primitives and hybrid runner

Added dual linear backward (primal/tangent input adjoints, combined weight gradient and primal-only bias), LayerNorm/sample-variance reverse-over-forward input and affine adjoints, and AdaLN shift/scale and gated-product adjoints. Shared conditioning requires caller-managed accumulation. Population LayerNorm uses the mean adjoint; sample-variance normalisation preserves its uncentred `x*alpha*scale` path and uses the N−1 denominator. Forward statistics are `[mean,dmean,inv,dinv,dvariance]` per row.

Added device dual forward linear, SiLU, shift/scale, product and normalisation/statistics operations. Vulkan has validated multi-stage plans for dual linear/norm gradients and dual norm forward. Kernels overwrite disjoint outputs and do not use float atomics. The flow tape integration below connects forward and reverse-over-forward operations to these primitives.

CUDA dual kernels are checked-in readable source (`backends/nvidia/ptx/training_dual.cu`) compiled by CUDA 12.8.93 to embedded PTX 8.7 for sm80, using `--fmad=false -ftz=false -prec-div=true -prec-sqrt=true --std=c++17`. This is a separate owned module and requires a compatible driver. It is never silently substituted with the earlier PTX7 module. Offline compilation/assembly and GLSL/SPIR-V validation pass. All driver execution remains unrun.

Dual norm host schedules matched independent F64 scalar-objective central differences across 144 values: maximum absolute error 1.8496625220354446e-7, RMS 2.7333309842228444e-8. Batched dual linear's 24 values had maximum error 2.9802373902043655e-8, RMS 8.28614031058571e-9. Tiny exact integer and independent directional-forward checks cover tangent bias, modulation and activation derivatives. Device output error has not been measured.

New scoped gates: 110 top-level passes for dual backward/admission/ownership (11 tests repeated ten times), race, vet and cross-builds in **20:50:25.317589774–20:50:39.656714369 UTC**; norm forward 40 passes plus race in **20:55:32.220688223–20:55:47.939709601 UTC**; other dual forward 40 passes plus race in **21:02:27.523432419–21:02:43.689071454 UTC**. New composed-plan, batched-linear and error-path tests also passed. Focused coverage is 100% for PTX dual wrappers and dual admission helpers; Vulkan dual stages are 85.7–90.9%, and forward-plan coverage is 76.9%. These are statement-coverage scopes, not numerical/device completeness; the selected-package aggregate is 8.7% because unrelated paths are excluded. Missing branches/native successful construction remain open. A dual-formula delegate timed out after 60 seconds with no review.

Vulkan constructor rollback now returns a partial owner when cleanup fails, preserving handles for explicit disposition. PTX two-module construction does likewise if its first module cannot be closed. CLI errors retain partial/terminal owners; the standalone CLI exits on error. No failed-cleanup retry or crash recovery is attempted.

### Training CLI hybrid scope

`ptx-attention-hybrid` and `vulkan-attention-hybrid` are now wired into the training runner behind `-allow-experimental-device-training` and a required caller-declared `-device-runtime-sha256`. Native setup follows option/local input admission. Replay metadata initially admitted `simd`, `ptx-attention-hybrid-v1` and `vulkan-attention-hybrid-v1`; the flow scope extension below adds two separate identities. The live backend must match at construction, update, checkpoint and restore. Options identity additionally binds the device/runtime digest. Full `ptx`/`vulkan` remains rejected.

Five runner/replay/CLI tests passed ten times (50 top-level passes), plus race, vet and host/ARM64/RISC-V CLI builds in **21:07:38.631132606–21:07:54.349446589 UTC**, exit 0. Synthetic named reference backends exercise actual tiny training, checkpoint/resume and cross-backend rejection; no GPU runtime was opened. Existing SIMD checkpoint format and identity are unchanged. No native adapter success/resource/performance acceptance follows.

Evidence: `/workspace/exports/pocket-gpu-training-dual-20261001/`. Prior gates and the unchanged eight-path inference index are preserved.

## Flow tapes and explicit flow hybrids

Flow forward/JVP and reverse-over-forward tape orchestration now passes the pinned host PyTorch mixed-flow, minimal-LSD and full-step fixtures with unchanged tolerances. `ptx-flow-hybrid` and `vulkan-flow-hybrid` have separate opt-in CLI factories and v1 replay identities. Transformer, time embeddings and scalar adjoints, loss reductions, clipping and AdamW/EMA stay on CPU. Parameters transfer per tape; no persistent model residency or device performance measurement exists.

The final offline runner gate at 21:40 UTC gave 80 repeated top-level passes, 80 subtest passes, focused race, vet, host build and Linux ARM64/RISC-V cross-builds. Hybrid backward failures stop retry and checkpoint publication before host parameter mutation. Restore also rejects a changed live scope. Native construction/execution coverage is limited; independent review timed out. Details and exact coverage are in [the flow-hybrid record](pocket-tts-flow-hybrid-20261001.md).

## RoPE and causal attention tapes

Adjacent-pair forward/transpose RoPE and exact-key causal attention forward now compose with the existing five-stage backward in a bounded Q/K/V tape. Host tests match an independent F64 objective for all input gradients and the existing CPU transformer convention. RoPE's 2,216 forward/adjoint values have maximum error 3.58e-7; forward attention's 4,605 probability/output values have maximum error 2.95e-7. Final offline repeated/race, compiler, vet, host/cross-build and layout gates pass; no native device calls ran. Whole-tape kernel preflight rejects missing later-stage owners before allocation. [The attention-tape record](pocket-tts-attention-tape-20261001.md) has commands, coverage and limitations.

## Complete stateless transformer tape

GELU-tanh and row gather/scatter/scale-reduction primitives now complete the transformer graph. The host tape matches pinned PyTorch transformer, FlowLM and full-step fixtures with unchanged tolerances, including all 12 tiny transformer parameter gradients. Separate `ptx-transformer-hybrid` / `vulkan-transformer-hybrid` CLI and replay scopes offload transformer forward/VJP; conditioner/EOS/flow and AdamW/EMA remain CPU. Final offline gate gives 180 repeated top-level passes plus race/vet/host/cross-builds. Parameters transfer twice per update. [The transformer record](pocket-tts-transformer-training-20261001.md) covers measured coverage, corrected GELU tail cancellation and open native/resource gates.

## Resident optimiser state

Explicit PTX/Vulkan drivers now support bounded resident parameter/gradient/M/V/EMA buffers, named branch accumulation, stable chunk norm summaries, AdamW/EMA and completed-boundary host snapshots. Host coordinator/mocked backend gates pass: 130 repeated tests, 70 subtests, race/vet/build/layout, plus 20 new boundary/inventory passes. The final norm combines small device summaries on CPU; graph parameters still transfer per tape and the runner is not yet bound to resident updates. [The resident-state record](pocket-tts-resident-optimizer-20261001.md) has ownership rules, numerical results, coverage and remaining integration.

## Resident tape bindings and optimiser runner

Named flow/transformer tape bindings now borrow resident parameters and accumulate gradients without host downloads. Active borrows block update/snapshot/close; uncertain tapes retain owners. Separate `ptx-optimizer-hybrid` / `vulkan-optimizer-hybrid` runner scopes use CPU gradients, resident device updates and coherent full host snapshots before replay commit/checkpoint. Final offline gate gave 130 repeated tests plus 130 subtests, race, 62 affected regressions and vet/build/layout. [The binding/runner record](pocket-tts-resident-bindings-20261001.md) covers scope, lifetime and measured coverage. Combined resident graph execution is not yet exposed.

## Combined resident graph runner

Separate `ptx-resident-hybrid` / `vulkan-resident-hybrid` modes now bind resident flow/transformer parameters, accumulate device gradients and apply resident AdamW/EMA under one replay owner. Endpoint parameter stop-gradient routing is explicit; CPU conditioner/EOS/weighting/loss and host norm reduction remain. Full host snapshots synchronise completed steps. Host tests compare all gradients and optimiser state with pinned PyTorch and cover update/resume/failure boundaries; 90 repeated tests plus race/regressions/vet/build/layout pass. [The combined record](pocket-tts-resident-combined-20261001.md) details scope, byte ceilings and coverage.

## Resident learned weighting

Separate `ptx-resident-weight-hybrid` / `vulkan-resident-weight-hybrid` scopes add the affine/ReLU learned weighting MLP and resident bias/weight gradients. Forward disables accumulation; one seeded backward accumulates parameters without downloads. Original resident-v1 retains CPU weighting. Independent F64 and pinned complete-step/PyTorch tests pass; 120 repeated tests plus race/regressions/vet/build/layout, followed by 20 ownership/topology passes. [The weighting record](pocket-tts-resident-weighting-20261001.md) covers scopes, coverage and remaining native/resource gates.

## Resident speaker/audio/EOS projections

Separate resident-projection v1 scopes now bind speaker/audio projections and EOS affine parameters, with direct device gradients and returned voice/audio/BOS derivatives. Token embeddings and sequence assembly stay CPU. Pinned complete-step/all-gradient/update parity passes unchanged; 100 repeated tests plus race,63regressions,vet/build/layout include empty-voice and failure/resume/identity gates. [The projection record](pocket-tts-resident-projections-20261001.md) has ownership rules and coverage.

## Resident embedding/BOS/sequence

Separate resident-sequence v1 scopes now execute embedding gather, sequence-order repeated-token VJP and BOS/prefix/shifted-audio assembly on tapes. All trainable parameter gradients accumulate on device; scalar losses/time embeddings and final global norm still run CPU with transfer-inclusive tape boundaries and full host snapshots. Pinned complete-step/all-gradient/update parity passes unchanged; 130 repeated tests plus50subtests,race,70regressions,vet/build/layout and 90 follow-up passes cover empty boundaries, shifting, ownership and failures. [The sequence record](pocket-tts-resident-sequence-20261001.md) describes deterministic kernels and remaining costs.

## Resident time embeddings

Separate resident-time v1 scopes now compute sinusoidal values/JVPs and reverse-over-forward scalar time adjoints inside flow tapes. Earlier scopes retain host trigonometry. New native kernels compile offline; host independent F64/finite-difference and unchanged mixed-AD/full-step fixtures pass.120repeated+20subtests,race,41regressions,vet/build/layout; mathematical delegate found no formula bug but reviewed no source/hardware. [The time record](pocket-tts-resident-time-20261002.md) distinguishes host rounding evidence from unqualified native trigonometry.

## Resident EOS loss

Separate resident-eos v1 scopes now compute stable BCE/weighted logit gradients and compensated scalar mean inside unbound utility tapes; legacy scopes preserve CPU F64 EOS. Explicit same-owner/failure retention and first-invalid/active-count policy hold. Independent F64/series checks and unchanged pinned full-step gradients/updates pass;150repeated+190subtests,race,36regressions,vet/build/layout. [The EOS record](pocket-tts-resident-eos-20261002.md) distinguishes loss rounding from native numerical qualification.

## Resident LSD losses

Separate resident-loss v1 scopes now compute normalized diagonal/minimal-LSD losses, compensated squares and prediction/logvar seeds on tapes. Endpoint input-only VJP and one-time objective scaling remain explicit. Independent F64/finite-difference and pinned fullstep/update pass unchanged;130repeated+110subtests,race,41regressions,vet/build/layout. [The loss record](pocket-tts-resident-loss-20261002.md) documents logvar semantics, overflow bounds and native limitations.

## Resident row branch arithmetic

Separate resident-branch v1 scopes now execute interpolation/advancement, minimal-LSD mixed seeds/target/noise/time adjoints and selected scaling on existing vector/affine tapes. CPU step accumulation/scalars/metrics still remain. Independent F64/small references and pinned minimal-LSD/fullstep/update pass unchanged;110repeated+180subtests,race,36regressions,vet/build/layout. [The branch record](pocket-tts-resident-branch-20261002.md) records reassociated diagnostic reductions and added transfer boundaries.

## Resident step reductions

Separate resident-step v1 scopes now combine condition/direct-target branches, scale logvar seeds, reduce F32 row metrics and add final target input gradients on utility tapes. CPU dispatch/boundary assembly and global norm remain. Independent F64 and pinned metrics/fullstep/update pass unchanged;120repeated+110subtests,race,37regressions,vet/build/layout. [The step record](pocket-tts-resident-step-20261002.md) documents weighting, padding and transfer limits.

## Resident tape dependency pruning

Separate resident-pruned v1 scopes now admit full source/inventory then execute only requested-output/accumulated-gradient closures, preserving command order. Pure forward derivatives use owned zero placeholders; seeded endpoint VJP stays live. Tiny step commands5746→3891,tensors7244→4996,logical tape bytes212136→142880,downloads148→116; native peak/latency unmeasured.210repeated+300subtests,race,41regressions,vet/build/layout with constant-only adapter mocks and retained owners. [The pruning record](pocket-tts-resident-pruned-20261002.md) describes remapping and remaining host construction costs.

## Device global norm and clipping

Separate resident-globalnorm v1 scopes find the global maximum then accumulate fixed-scaled squares with persistent compensation and compute clipping on device. Four diagnostic/scale scalars transfer; legacy F64 reduction unchanged. Initial4096-parameter rescale drift fixed without tolerance changes.210repeated+180subtests,race,114regressions,clippedmultistep,vet/build/layout and offline CUDA/PTX/SPIR-V pass. [The global norm record](pocket-tts-resident-globalnorm-20261002.md) covers overflow/subnormal and host handoff limits.

## Resident device-scale handoff

Separate resident-devicescale v1 scopes now pass the resident clip buffer directly to a separate AdamW kernel, with no downloaded scale returned through launch constants. Complete-pass metadata preflight precedes reduction; four diagnostic scalars still validate before mutation. Legacy globalnorm preserves its host constant path; residency remains 48 extra logical bytes. Targeted repeated/race, boundary failure follow-up, 144 regressions + 150 subtests, builds and offline compilation passed. [The device-scale record](pocket-tts-resident-devicescale-20261002.md) documents the remaining diagnostic synchronization and unqualified native behavior.

## Resident weighting sample packing

Separate resident-sampleweight v1 scopes pack frozen time columns and already-scaled logvar seeds inside the weighting MLP tape, removing host duplicate-time matrix construction and seed concatenation without another objective scale. Legacy scopes retain CPU packing. Independent F64 forward/VJP, full-step/clipped multistep/resume, ownership and seeded/pruning gates pass: 170 repeated + 110 subtests, race, 99 regressions + 107 subtests, vet/build/crossbuild/layout. Tiny commands 3,891→3,958, logical bytes 142,880→144,560, downloads 116 unchanged; no native performance improvement claimed. [The sample-weight record](pocket-tts-resident-sampleweight-20261002.md) documents the added tape costs and remaining boundaries.

## Resident host-state copy reduction

Full-state admission now borrows arrays read-only; owned conversion/load helpers retain their copying contract. Resident graph/optimiser boundary checks compare live state against the owned snapshot without cloning `State()`. Tiny boundary allocations 275→4; the 32,768-element validation fixture removes discarded copy work (525,440→1,152 B/op), without changing device snapshots/downloads or replay arithmetic.60repeated+170subtests,race,109regressions+129subtests,vet/build/crossbuild/layout pass. [The state-copy record](pocket-tts-resident-statecopies-20261002.md) documents the limits of this host-allocation evidence.

## Resident snapshot ownership handoff

Completed resident updates privately consume freshly owned snapshot arrays into the backend boundary, avoiding a second parameter/M/V/EMA clone. Public conversion and live trainer ownership remain independent; full device snapshots still transfer. Tiny conversion allocations 295→59;32,768-element conversion526,808→2,520 B/op.30repeated+110subtests,race,112regressions+140subtests,vet/build/crossbuild/layout pass. [The snapshot handoff record](pocket-tts-resident-snapshothandoff-20261002.md) documents trusted-source and terminal-load requirements; design-only judge review is not source/native approval.

## Resident live moment/EMA storage reuse

Private completed-update loading now validates the full state and disjoint source/destination ranges before copying into reusable live moment/EMA storage. Public restore and independent backend boundary ownership remain unchanged. Synthetic isolated allocation falls about1.13MB→47.5KB/op (276→30allocations),but no latency improvement established;initial duplicate scans corrected before rerunning gates.40repeated+350subtests,race,116regressions+175subtests,final guard follow-up/race,vet/build/crossbuild/layout pass. [The load-reuse record](pocket-tts-resident-loadreuse-20261002.md) preserves the regression/timing limits and full snapshot transfer boundary.

## Resident parameter-copy removal

Private synchronous resident builders borrow only named parameter constants read-only instead of cloning host weights before device substitution. Public builders still own copies; full source/binding validation, logical ceilings, commands and replay scopes remain unchanged. Isolated weighting-builder allocation302KB→34.9KB/op; no native request speedup claimed.20repeated+80subtests,race,135regressions+189subtests,vet/build/crossbuild/layout pass. [The parameter-copy record](pocket-tts-resident-paramborrow-20261002.md) documents source lifetime and remaining graph materialisation.

## Resident intra-tape lifetime storage

Explicit resident-lifetime v1 scopes reuse equal-sized intermediate allocations after the previous last consumer, retaining constants/borrowed parameters/declared-output reshape families. Complete source admission, dispatch disjointness and failure ownership remain enforced;legacy allocations unchanged. Tiny summed reshape-aware bytes93,368→63,776,owned excluding borrowed73,080→43,488;commands3958/downloads116 unchanged,not native peak/latency.230repeated+110subtests,race,185regressions+245subtests,refinedaccounting followup,vet/build/crossbuild/layout pass. [The lifetime record](pocket-tts-resident-lifetime-20261002.md) documents the unchanged transfers and native qualification limits.

## Resident device-merged state checks

Explicit resident-statecheck v1 scopes merge existing chunk finite/nonnegative mutable-state diagnostics on device before step commit,returning one four-scalar result. Full preflight/terminal retention and legacy perparam behavior remain. Tiny59param resident downloads473→238;update diagnostics237→2,full snapshots236 unchanged;no native latency claim.220repeated+260subtests,race,155regressions+263subtests,vet/build/crossbuild/layout pass. [The state-check record](pocket-tts-resident-statecheck-20261002.md) documents invalid/negativeV/second-phase failure gates and remaining sync.

## Resident graph-local zero sources

Private synchronous builders pool immutable literal-zero host sources by extent within one graph,without merging IDs/native storage or weakening logical ceilings. Public arrays and caller inputs/seeds retain ownership. Isolated64×32768F32 materialisation8.39MB→131KB/op,not complete-request/device evidence.40repeated+80subtests,race,161regressions+219subtests,vet/build/crossbuild/layout pass. [The zero-source record](pocket-tts-resident-zeropool-20261002.md) documents read-only/lifetime limits and remaining graph construction.

## Resident shared device zero constants

Explicit resident-zeroconst v1 scopes share pinned equal-size positive-zero constants;negativezero/boundparameter IDs excluded,writable storage/fullsource budget/legacyalloc unchanged. Tiny plannedowned43488→34956bytes,downloads unchanged;primitive mockzero allocations/uploads reduced,no nativepeak/speedclaim.230repeated+110subtests,race,210regressions+267subtests,vet/build/crossbuild/layout pass. [The zero-constant record](pocket-tts-resident-zeroconst-20261002.md) covers protected parameter identity,signedzero,terminalretention and native limits.

## Resident fused EOS boundary

Explicit resident-eosfused v1 scopes combine EOS affine/BCE/weightedVJP in one bound tape;step-local hiddenVJP replaces EOS backward reevaluation,single parameter accumulation/scaling retained. Standalone/legacypaths unchanged. Tiny tapes49→47/downloads116→113,commands3958→3959/full snapshots unchanged;no native latency claim.170repeated+150subtests,race,157regressions+227subtests,vet/build/crossbuild/layout pass. [The fused-EOS record](pocket-tts-resident-eosfused-20261002.md) covers independent VJP/metric/admission evidence and remaining boundaries.

## Resident fused shifted-audio boundary

Explicit resident-audiofused v1 scopes combine BOS/shift assembly/inputprojection VJPs;no shifted host input,oneBOS+weightaccumulation/finalnormalizedrowzero retained. Tiny tapes47→45/downloads113→109,commands3959→3962/full snapshots unchanged.190repeated+170subtests,race,175regressions+244subtests,vet/build/crossbuild/layout pass. [The fused-audio record](pocket-tts-resident-audiofused-20261002.md) documents independent affine/BOS VJP references,ownership/admission and native limits.

## Resident fused speaker/sequence boundary

Explicit resident-voicefused v1 scopes combine rawspeakerprojection/sequence VJP;projectedvoice boundaries removed,threeparamgradients once,emptyvoice zero contribution/repeatedtokens preserved. Tiny tapes45→43/downloads109→105,commands3962→3942/full snapshots unchanged.210repeated+210subtests,race,193regressions+265subtests,vet/build/crossbuild/layout pass. [The fused-voice record](pocket-tts-resident-voicefused-20261002.md) documents independent speaker/sequence/VJP and native limits.

## Resident combined conditioner boundary

Explicit resident-conditioner v1 scopes combine rawspeaker/text/shiftedaudio sequenceVJPs;projectedaudio boundary removed,fiveparamgradients once,rawinput/finalrowzero/emptyvoice/repeatedtokens preserved. Tiny tapes43→41/downloads105→103,commands3942 unchanged/full snapshots unchanged.230repeated+230subtests,race,214regressions+288subtests,vet/build/crossbuild/layout pass. [The conditioner record](pocket-tts-resident-conditioner-20261002.md) documents independent complete VJP and native limits.

## Resident conditioner–transformer boundary

Explicit resident-backbone v1 scopes connect conditioner forward, transformer forward/VJP and conditioner backward through one bound graph. Public sequence output, all parameter bindings and legacy paths are preserved. Tiny tapes 41→39 and downloads 103→101; commands 3,942→3,967 and logical bytes 141,784→144,112. Repeated/race, pinned PyTorch/all-gradient, host-reference, resume/clipping and build gates pass. Native performance and independent source approval are open. [Backbone validation](pocket-tts-resident-backbone-20261002.md) records the tradeoff, CLI factory correction and remaining handoffs.

## Resident EOS–backbone backward boundary

Explicit resident-eosbackbone v1 scopes retain scalar-only forward EOS loss/admission and combine weighted EOS hidden VJP with flow condition seeds inside backbone backward. EOS parameters accumulate once. Tiny downloads 101→100, tapes 39 unchanged; commands 3,967→3,980 and logical bytes 144,112→145,328. Repeated/race, independent composition/all-gradient, resume/clipping and build gates pass. Native performance and source approval are open. [EOS/backbone validation](pocket-tts-resident-eosbackbone-20261002.md) records arithmetic and retention checks.

## Resident diagonal loss–flow backward boundary

Resident-diagfused v1 retains initial prediction overflow admission and combines loss seeds/objective scaling/flow VJP/target gradient. Raw stats/logvar scaling and all-binding single accumulation are preserved. Tiny two-valid-row tapes 39→33, downloads 100→86, commands 3,980→3,956. Repeated/race, independent CPU/F64/composed references, pinned full-step, clipping/resume and builds pass. [Diagonal fusion validation](pocket-tts-resident-diagfused-20261002.md) records boundaries and native/review limits.

## Resident distillation endpoint boundary

Resident-endpointfused v1 retains forward prediction overflow admission and combines loss/negated seed/endpoint input-only VJP/mixed primary seeds. No endpoint parameter/condition/time contribution accumulates; later primary/step objective scaling is unchanged. Tiny tapes 33→27, downloads 86→72 and commands 3,956→3,716. Independent CPU/F64/composed references, repeated/race, pinned full-step/LSD and resume/clipping/build gates pass. [Endpoint fusion validation](pocket-tts-resident-endpointfused-20261002.md) records remaining boundaries and review/native limits.

## Resident primary distillation finish boundary

Resident-primaryfused v1 combines mixed flow backward and target/noise/time finish operations. Raw input derivatives remain scale-independent; parameters accumulate with the objective scale once. Tiny tapes 27→25, downloads 72→62 and commands 3,716→3,694. Independent mixed CPU/F64/composed references, repeated/race, pinned full-step/LSD and resume/clipping/build gates pass. [Primary fusion validation](pocket-tts-resident-primaryfused-20261002.md) records remaining boundaries and review/native limits.

## Resident combined distillation backward boundary

Resident-distillfused v1 connects endpoint input-only VJP and primary mixed/finish gradient edges. Live parameters share identity; only primary gradients accumulate with objective scale once. Raw derivatives/forward overflow admission and legacy paths are preserved. Tiny tapes 25→23, downloads 62→54, commands unchanged. Independent CPU/F64/composed references, repeated/race, pinned full-step/LSD and clipping/resume/build gates pass. [Distillation fusion validation](pocket-tts-resident-distillfused-20261002.md) records remaining boundaries and source/native limits.

## Resident distillation forward boundary

Resident-forwardfused v1 connects interpolation/primary flow-JVP/advance/endpoint through device input edges, with no forward accumulation. Six public/overflow-admission values remain. Tiny tapes 23→17, downloads 54→50, commands 3,694→3,688. Independent CPU/composed references, repeated/race, pinned full-step/LSD and clipping/resume/build gates pass. A mathematical judge confirmed the stated composition invariants; source/native approval is open. [Forward fusion validation](pocket-tts-resident-forwardfused-20261002.md) records the remaining host boundaries.

## Resident diagonal forward boundary

Resident-diagforward v1 connects diagonal interpolation and initial flow through a device input edge, retaining three owned prediction/admission outputs and no forward accumulation. Tiny tapes 17→15, downloads 50→48, commands 3,688→3,684. Independent CPU/composed references, repeated/race, pinned full-step and clipping/resume/build gates pass. [Diagonal forward validation](pocket-tts-resident-diagforward-20261002.md) records shared-helper coverage and source/native limits.

## Resident step-only weighting roots

Resident-weightroots v1 accumulates all sample-weight parameter gradients without the training step's discarded forward/input-VJP downloads. Public seeded APIs and full-source admission are unchanged. Tiny downloads 48→46, tapes 15 unchanged and commands 3,684→3,680. Independent F64/captured-gradient, repeated/race, pinned full-step and clipping/resume/build gates pass; new-helper coverage is 100%. [Weight-root validation](pocket-tts-resident-weightroots-20261002.md) records empty-output completion and source/native limits.

## Resident step reduction–weighting seed boundary

Resident-stepweight v1 keeps scaled logvar seed edges inside the composed reduction/weighting tape, with all public reduction outputs and once-only scaling/accumulation preserved. Tiny tapes 15→14, downloads 46 and commands 3,680 unchanged. Independent F64/composed/captured-gradient, repeated/race, pinned full-step and clipping/resume/build gates pass. [Step/weighting validation](pocket-tts-resident-stepweight-20261002.md) records admission, coverage and source/native limits.

## Resident reduction–backbone backward boundary

Resident-stepbackbone v1 composes reduction/weighting/EOS-backbone backward/final direct-target addition. Public reduction outputs and all parameter roots remain; scaled condition/logvar edges staydevice. Tiny tapes 14→12, downloads 46→45, commands 3,680→3,682. Pinned full-step/all-gradient, references, repeated/race, clipping/resume and builds pass. [Step/backbone validation](pocket-tts-resident-stepbackbone-20261002.md) records combined admission, coverage and source/native limits.

## Resident private final-step result selection

Resident-stepresults v1 omits discarded condition/direct-target full downloads while retaining logvar/metrics/input gradients and an exact direct-tail witness. Public APIs/parameter roots/scaling are unchanged. Tiny downloads 45→44, tape-result elements 190→174 and tapes 12 unchanged. Repeated/race, pinned full-step/references, clipping/resume/build gates pass; new-helper scoped coverage is 100%. [Selected-result validation](pocket-tts-resident-stepresults-20261002.md) records the one-command guard tradeoff and native/source limits.

## Resident private distillation results

Resident-distillresults v1 selects step-only raw condition/target/statistics plus a scalar finite witness for discarded noise/time derivatives. Public LSD outputs, primary-only gradient roots and once-only scaling are unchanged. Tiny downloads 44->42, elements 174->168, tapes 12 unchanged; the witness adds fourteen commands and 152 logical bytes. Repeated/race, pinned references/full-step/clipping/resume and build gates pass; all five new helpers have 100% scoped coverage. [Private distillation-result validation](pocket-tts-resident-distillresults-20261002.md) records finite propagation, accounting, development corrections and native/source limits.

## Resident combined backbone/EOS forward

Resident-eosforward v1 evaluates read-only backbone forward and EOS mean in one tape. Sequence/hidden/scalar boundaries stay owned; the same host absolute-dot bound gates all gradients after download. Public FlowLM dispatch and EOS weight-once backward are unchanged. Tiny tapes 12->11, downloads 42/elements 168 unchanged, commands 3697->3700 and planned owned bytes 27960->27864. Repeated/race, pinned full-step/references, clipping/resume and build gates pass; builder coverage 96.2%, decoder/evaluator 100%. [EOS-forward validation](pocket-tts-resident-eosforward-20261002.md) records admission timing and unmeasured native costs.

## Fused-boundary coverage follow-up

Four existing fused component evaluators now decode through private count/shape/finite/ownership helpers. All four decoders and diagonal/primary/distillation/forward/stepweight evaluators reach 100% scoped coverage; backbone builder reaches 98%, and stepbackbone decoder 100%. Distillation/forward builder coverage measured 90.5%/87.2% in that gate. The subsequent [composition-admission follow-up](pocket-tts-resident-composition-20261002.md) extracted private source-composition helpers and closed both files at 100% scoped coverage; command order/scaling/public APIs/scopes are unchanged. [Coverage validation](pocket-tts-resident-coverage-20261002.md) lists the remaining fixed-builder assertions, failure-retention tests and combined baseline/final selectors. Arithmetic, public results and replay identities are unchanged.

## Resident selected forward results

Resident-forwardresults v1 returns private audio hidden/EOS mean/finite witness outputs, omitting discarded sequence/transformed-prefix downloads. Public FlowLM outputs, overflow admission and all gradient roots/scales are unchanged. Tiny tape-result elements 168->117, tapes 11/downloads 42 unchanged; commands 3700->3709 and planned owned bytes 27864->27876. All five new helpers and bounded finite-witness helper have 100% scoped coverage. [Forward-result validation](pocket-tts-resident-forwardresults-20261002.md) records chunk ceilings, read-only failures and unmeasured native costs.

## Private LSD diagnostics

The resident step-only LSD call skips four discarded host diagnostic arrays after executing the same admitted forward/backward graphs. Public/legacy outputs and all scales/roots/scopes are unchanged. Five bounded tiny mock samples show median private allocations 9004->9000, with graph construction still about 2.85 MB/op and no latency improvement established. [Diagnostic-copy validation](pocket-tts-resident-lsddiagnostics-20261002.md) records public/private parity, ownership, failure retention and measurements.

## Composition metadata reservation

Distillation forward/backward composers reserve final bounded tensor/command/output metadata once after complete source admission. IDs/order/arithmetic/ownership/scopes are unchanged. Alternating tiny mock samples reduce private complete-row allocation 2.854->2.479 MB/op (13.1%), about 22 allocations; timing ranges overlap. Re-profiled composition metadata hotspots fall about 49--52%; pruning/initial graph growth remain. [Host-graph validation](pocket-tts-resident-hostgraph-20261002.md) records profiles, retained-memory limits, combined-ceiling tests and native gaps. Both modified source files retain 100% scoped coverage.

## Pruning metadata reservation

`PruneTape` counts the admitted closure and reserves exact owned result metadata once, preserving root/remap order and read-only values. Isolated 1024-command allocation falls 401224->205976 B/op (48.7%); private complete tiny mock row 2.479->2.100 MB/op (15.3%). Attribution falls 61.5% for pruning; timing ranges overlap. [Pruning-metadata validation](pocket-tts-resident-prunemetadata-20261002.md) records 97.9% scoped coverage, retention/profile limits, backend tests and the existing unrun MPFR diagnostic. Numerical contracts/scopes/accounting are unchanged.

## Flow command metadata reservation

Flow builders reserve bounded command metadata at forward entry and before unchanged reverse emission, counting adjoint-presence joins without executing math. Private complete tiny mock allocation falls 2.100->1.885 MB/op (10.2%), 28 allocations; timing ranges overlap. Re-profiled flow-builder cumulative allocation falls about 22.2%, with no post-GC increase. Three new helpers have 100% scoped coverage. [Flow-metadata validation](pocket-tts-resident-flowmetadata-20261002.md) records rejected reservations, count tests, retained-memory/native limits and the unrun MPFR diagnostic. IDs/order/tensors/scales/ownership/scopes are unchanged.

## Pruning metadata arena

Pruned shape/operand lists use per-call 256-int chunks with disjoint cap-limited slices; no mutable cache or source borrowing change. Isolated 1024-command pruning allocations fall 2061->24 with identical bytes. Private complete tiny mock row allocations fall 8902->6641 (25.4%), total bytes nearly unchanged. [Arena validation](pocket-tts-resident-prunearena-20261002.md) records append/concurrency isolation, bounded incidental retention, tiny size-class byte costs, profiles, helper100%/Prune98.1% coverage and the unrun MPFR diagnostic. Numerical contracts/roots/scopes/accounting are unchanged.

## Flow shape metadata arena

Flow tensor shapes use copied disjoint cap-limited per-graph chunks, preserving public mutation/append isolation and source admission. Private complete tiny mock allocations fall 6641->4508 (32.1%); bytes rise about 6.3 KB/op (0.33%) from padding. Tensor-object attribution falls about 96.5%, with no observed post-GC increase. [Shape-arena validation](pocket-tts-resident-flowshapes-20261002.md) records bounded incidental retention, concurrent-build tests, helper100% coverage and unmeasured native costs. IDs/order/arithmetic/scopes are unchanged.

## Flow command operand arena

Flow commands copy operand lists into disjoint cap-limited per-graph chunks, isolating sibling command writes/appends. Private complete tiny mock allocations fall 4508->3068 (31.9%); bytes rise about 17.7 KB/op (0.94%) from padding and removing shared reverse-stage operand lists. Attribution objects fall about 26.5%, post-GC unchanged. [Operand-arena validation](pocket-tts-resident-flowoperands-20261002.md) records ownership/concurrency tests, helper100% coverage, byte/retention tradeoffs and native limits. IDs/order/math/admission/scopes/accounting are unchanged.

## Composition operand storage

Distillation/forward remappers now use local bounded operand-copy arenas instead of allocating each remapped list. Full-source admission, IDs/order/scales/roots, source borrowing and independent mutable remapped slices are unchanged. Private complete tiny mock allocations fall 3068->1781 (41.9%); allocated bytes rise about 1.5 KB/op (0.08%). Bounded attribution objects fall 33.7%, with post-GC totals essentially unchanged. [Composition-operand validation](pocket-tts-resident-compositionoperands-20261002.md) records mapping/ownership/concurrency/retention tests, 100% composition-file scoped coverage and byte costs. Source review timed out; invariant review alone does not establish native performance or approval.

## Inline flow AD-node inputs

Private flow reverse nodes now hold their unary/binary operands in a fixed two-slot value array, removing separate input-list allocations. Reverse emission/counting, IDs/order/scales/public tapes and admission are unchanged. Private complete tiny mock allocations fall 1781->1625 (8.8%) and bytes fall about 2.8 KB/op (0.15%). Nodes grow eight bytes on amd64 but avoid 16/32-byte backing arrays; isolated timing ranges overlap. [Flow-node validation](pocket-tts-resident-flownodes-20261002.md) records metadata copy ownership, all-node-kind/count/race/numerical gates, profile and byte tradeoffs. Invariant review is not independent source/native approval.

## Indexed embedding VJP primitive

Embedding mode two accepts fully validated occurrence offsets/row indices and sums only matching rows in sequence order, without atomics. Dense output/accumulation remains; existing Pocket TTS builders, CLI and replay routes still use modes zero/one. [Indexed-embedding validation](pocket-tts-resident-embeddingindex-20261002.md) records constant admission, independent F64/order/pruning references, mock ABI/terminal guards, offline PTX/SPIR-V compilation and index setup costs (about 410 KB host allocation and 133 KB packed index for vocabulary32768/rows512). No native performance or source approval is claimed. Resident integration and removal of dense gradient work remain open.

## Resident indexed embedding integration

Explicit `ptx-/vulkan-resident-embeddingindex-hybrid-v1` scopes inherit selected private results and rewrite admitted mode-one embedding VJPs to indexed mode two before pruning/allocation. Public builders and legacy scopes retain mode one; replay hash kind is `resident-embeddingindex-backend`. [Resident index validation](pocket-tts-resident-residentembeddingindex-20261002.md) records full-step/all-gradient/update/EMA/clipping, repeated/empty text, replay, owner/failure and late-index-budget tests. Tiny tapes11/downloads42/commands3709 remain; logical bytes134064->134100 and planned storage41984->42020. Dense gradients/accumulation and host handoffs/snapshots remain; index costs are additional, not a native-speed claim. Invariant judge timed out; no source/native approval.

## Compact embedding VJP primitive

Embedding mode three materialises sequence-ordered active-token slot gradients padded to rows, rather than a vocabulary-sized table. Current CLI/resident routes and binding rules are unchanged: compact gradients need a separate accumulation contract. [Compact-embedding validation](pocket-tts-resident-embeddingcompact-20261002.md) covers index admission, F64/order/padding/pruning references, mock ABI/terminal guards and offline compilation. At vocabulary32768/width1024/rows512, theoretical output falls128MiB->2MiB and packed index133124->6148B; host setup10624B/two allocations versus409600B, but compact setup is not faster in the measured fixture. No native/source approval or full-device claim; resident gradient buffers/optimiser work remain.

## Compact embedding accumulation primitive

An optional PTX/Vulkan driver interface accumulates compact slot gradients into dense ping-pong resident storage using sorted-token lookup and the unchanged separate previous+scale*contribution formula. It avoids dense tape expansion only when compact bindings are subsequently integrated; current coordinator/CLI routes and ordinary binding extents remain unchanged. [Compact accumulation validation](pocket-tts-resident-embeddingaccumulate-20261002.md) records preflight/extents/alias/context/kernel guards, mock submit/sync terminal retention and F64 references. Dense buffers/work remain; binary lookup may add arithmetic. Operator construction adds one PTX entry and Vulkan pipeline27->28 even for older scopes. Native performance/source approval remain open; judge timed out.

## Compact resident binding/coordinator contract

A separate `RunCompactZeroPrunedTape` API admits explicit compact descriptors with direct mode-three gradient/index provenance and full constant-index checks. Ordinary dense bindings reject compact outputs/reshape aliases even when extents match. Requested outputs, all gradients and compact indices survive pruning; the complete dense/compact accumulation pass is preflighted before exactly-once ping-pong mutation. [Compact binding validation](pocket-tts-resident-compactbindings-20261002.md) records mixed-gradient arithmetic, repeated scales, borrow guards, early inventory/driver rejection and failure retention. No CLI/replay scope selects it yet; dense persistent buffers/work remain. Invariant judge found no concrete blocker, not source/native approval.

## Resident compact embedding integration

Explicit `ptx-/vulkan-resident-embeddingcompact-hybrid-v1` scopes add compact VJP/index roots only after complete dense-source admission, then use the compact coordinator and accumulation primitive. Private pruning removes the dense VJP temporary; explicit public dense-gradient downloads remain numerically intact and accumulate compact once. Older scopes/public builders stay unchanged; replay hash kind is `resident-embeddingcompact-backend`. [Resident compact validation](pocket-tts-resident-residentembeddingcompact-20261002.md) covers fullstep/updates/clipping/repeat/emptytokens, actualdispatch/publicdense, capability/budget/failures and resume/identity. Tiny11tapes42downloads3709commands remain/storage41984->41948; synthetic16x16/vocab4096 pruned tape263232->2244B while fullsource525376->526596B. Dense persistent gradient/optimiser buffers/work remain; no native peak/speed/source approval claim.

## Scalar LSD prediction-admission primitive

LSD-loss mode two reports a scalar invalid flag without loss/gradient work. `LSDPredictionCutoff` binary-searches the largest F32 magnitude satisfying the original host F64 square/seed-overflow comparisons; tape admission binds the cutoff to logvar and kernels check finite inputs/F32 differences. [Scalar admission validation](pocket-tts-resident-lsdadmission-20261002.md) records cutoff/nextafter/sign/subtraction/runtime-NaN boundary tests, mock ABI/terminal checks and offline compilation. Existing modes/CLI/replay routes and prediction downloads remain unchanged; integration before backward accumulation is follow-up. Native FTZ/resources/performance and source approval remain open; judge timed out.

## Resident diagonal scalar admission

Explicit `ptx-/vulkan-resident-diagadmission-hybrid-v1` scopes inherit compact embedding and append the scalar prediction guard to private read-only diagonal forward. The step passes known logvar; flag/ownership/vector checks and redundant original host admission precede diagonal backward. Public/legacy calls and prediction downloads stay unchanged. [Diagonal admission validation](pocket-tts-resident-diagadmission-20261002.md) records ordering/failure/pinned-update/replay gates and costs: tiny11tapes unchanged, commands3709->3711/downloads42->44/elements117->119/storage41948->41964. Per-row gate, not whole-step gradient atomicity. Distillation/handoffs and native FTZ/source approval remain open.

## Resident distillation scalar admission

Explicit `ptx-/vulkan-resident-distilladmission-hybrid-v1` scopes add private step-only scalar admission on endpoint prediction/desired before endpoint/primary VJP, inheriting diagonal guard/compact embedding. Six public forward vectors, prediction downloads and redundant host guard remain; public LSD route is unchanged. [Distillation admission validation](pocket-tts-resident-distilladmission-20261002.md) records fullstep/clipping/replay/ownership and later-row failure tests: no completed update/metrics/replay/checkpoint/retry, while earlier resident gradients may exist. Tiny11tapes unchanged, commands3711->3713/downloads44->46/elements119->121/storage41964->41980 versus diagonal-only baseline. No native speed/source approval; device handoffs remain.

## Read-only tape handoff primitive

`ResidentTapeHandoff` owns a producer/consumer pair through explicit external accumulation, binds declared producer outputs as read-only consumer constants without a producer download, and closes the consumer first. Complete graph/edge/kernel/binding admission and a combined logical-byte ceiling precede producer allocation; duplicate/reshape-alias producer edges fail early. Late failures retain owners and block retry/output/close; external accumulation errors use `RetainFailure`. The caller serialises all methods and borrowed-buffer use, including `Close` versus `RunDevice`; successful runs must be device-complete. [Handoff validation](pocket-tts-resident-tapehandoff-20261002.md) records 831 regression top-level +1,365 subtest passes, the existing MPFR skip, and 768 independent F64 comparisons with maximum error 5.49e-8 under the unchanged 2e-6 limit. This standalone primitive has no CLI/scope integration; training downloads, native qualification and independent source approval are unfinished.

## Resident producer lease

`ResidentTapeLease` pins one completed producer across multiple pruned consumers, preserving original-to-pruned edge IDs, current resident parameter binding, compact index/gradient roots and scale-once accumulation. Public state operations stay blocked while the lease is live; invocation borrowing guards re-entry without clearing the lease. Combined complete-source ceilings precede consumer allocation. Late row failures/cancellation retain producer/failed consumer and block update/snapshot/close/retry, while prior gradients remain. [Lease validation](pocket-tts-resident-tapelease-20261002.md) records 848 regression top-level +1,416 subtest passes, the existing MPFR skip, and 768 independent F64 comparisons with maximum error 3.48e-8 under unchanged 2e-6. The primitive is outside model/CLI routing; existing downloads/scopes and native holds are unchanged.

## Opt-in transformer-to-flow handoff

Explicit `ptx-/vulkan-resident-flowhandoff-hybrid-v1` scopes retain one transformer/EOS producer across private diagonal/distillation forward row consumers. Full-source reshape provenance/constant admission precedes appended read-only hidden-root/vector row slices; combined live ceilings and native binding preflight precede pruning/allocation. Other row tapes use authorised no-edge consumers; producer close precedes final reduction/backbone/update, with terminal fencing on late guard/cancel/close failures. Host hidden/EOS guards, prediction/backward downloads and public/legacy routes remain. [Integration validation](pocket-tts-resident-flowhandoff-20261002.md) records pinned loss/all-gradient/update/EMA/resume/scaling/owner gates, 872 regression top-level +1,460 subtests and existing MPFR skip. Tiny11tapes/46downloads unchanged, commands3713->3717/storage41980->42108; no native speed/memory qualification or independent source approval.

## Conservative EOS admission primitive

A scalar `eos-admission` tape command bounds all-row EOS absolute dots with integer mantissa ceilings and 2^95 units under the original F64 limit, reserving one unit for host accumulation rounding. PTX/Vulkan kernels avoid floating arithmetic/FTZ in the check and may conservatively reject near-limit host-safe inputs. [EOS admission validation](pocket-tts-resident-eosadmission-20261002.md) records 128 whole-row cases/zero unsafe accepts, 100,000 independent product checks, documented conservative margins and 882 regression top-level +1,460 subtest passes with existing MPFR skip. All new reference/wrapper functions reached 100% scoped coverage. No CLI/scope integration or guard/download removal; native/source approval open. Vulkan pipeline construction28->29 plus complete EOS/compact-owner cleanup; constructor overhead unmeasured.

## Opt-in producer EOS admission

`ptx-/vulkan-resident-eosadmission-hybrid-v1` scopes append a scalar absolute-dot guard to the retained transformer/EOS producer. Complete source/bindings and affine/reshape/slice provenance precede expanded admission and lease construction. Flag extent/finite-zero/alias/input ownership and redundant original host guard precede all gradients; failure fences publication/retry/cleanup. [Integration validation](pocket-tts-resident-residenteosadmission-20261002.md) records pinned full-step/update/EMA/clipping/resume/owner/ordering gates and 904 regression top-level +1,508 subtests with existing MPFR skip. Tiny11tapes unchanged, commands3717->3719/downloads46->47/elements121->122/logicalbytes134236->134256/storage42108 unchanged. Conservative near-limit rejection is scope-explicit; hidden/prediction/backward downloads remain, native/source approval open.

## Retained child producers

`BeginChildTape` registers retained read-only producers under a lease root; `RunSourcesTape` binds pinned roots from exact live owners in one consumer. All live complete-source charges join consumer admission; parent/sibling dependencies prevent premature close, and copied/foreign/native-alias owners fail admission. Full inventory, stable pruning/compact index roots, scale-once accumulation and terminal retention stay in the shared runner. [Child validation](pocket-tts-resident-tapechild-20261002.md) records 917 regression top-level +1,541 subtest passes with existing MPFR skip, and 128 independent F64 comparisons with maximum2.64e-7 under unchanged2e-6. No model/CLI integration or training-download removal; independent source/native approval open.

## Opt-in retained prediction handoff

`ptx-/vulkan-resident-predictionhandoff-hybrid-v1` retains each private diagonal/distillation forward child through backward execution/accumulation. Explicit input/desired/prediction/velocity/tangent read edges and transformer condition slices preserve adjoint commands, primary-only gradients and scale-once policy. Host guards/downloads remain; source/provenance/combined-live admission precedes allocation, child close precedes parent close, and late failure/cancel/close errors fence publication. [Prediction handoff validation](pocket-tts-resident-predictionhandoff-20261002.md) records pinned loss/all-gradient/update/EMA/replay and 941 regression top-level +1,585 subtests/MPFR skip. Tiny11tapes47downloads122elements161accumulations unchanged; commands3719->3707/plannedstorage42108->42268. Native memory/speed/source approval open.

## Shape-only external input contract

`TensorSpec.External` declares a read-only shape-only device source with no host values; complete binding admission includes unused externals before pruning/allocation. Native constructors borrow exact live buffers, pruning preserves declarations, and lifetime/zero planning pins external source families. Host-index/time/label/cutoff metadata still requires explicit values. [External input validation](pocket-tts-resident-externalinputs-20261002.md) records conversion ownership, missing/dead binding rejection, borrowed-only mock construction/close, 512 independent F64 comparisons max2.79e-8 under unchanged2e-6, and 953 regression top-level +1,585 subtest passes/MPFR skip. No model/CLI integration or download removal; source/native approval open.

## Private shape-only flow builder

The private reverse-over-forward flow builder declares four external device sources directly, shares topology/extent validation and graph assembly, and preserves host/device-time commands/IDs/parameter roots. Public builders and runtime scopes retain host numerical admission; no production caller or download removal is enabled. [Flow device-input validation](pocket-tts-resident-flowdeviceinputs-20261002.md) records retained-lease borrowing/scale-once, complete-source/pruning ownership, unchanged tiny 11,984-byte ceiling, 786 pinned PyTorch comparisons max1.97e-7 under existing tolerances, and 975 regression top-level +1,631 subtest passes/MPFR skip. Retained-owner/scalar-admission integration and independent source/native approval are unfinished.

## Private retained-device diagonal path

The private diagonal source helper captures exact retained roots/model/row/time and freezes arithmetic metadata, checks LSD cutoff plus a finite witness through two scalars before accumulation, then builds shape-only backward and closes the forward child before row publication. Single-attempt/terminal retention, target/scaling order, primary parameter inventory and complete live ceilings are preserved. [Diagonal device-input validation](pocket-tts-resident-diagonaldeviceinputs-20261002.md) records 27 objective/time/logvar combinations, 10,530 independent CPU/F64 comparisons max2.38e-7 under unchanged gates, and 985 regression top-level +1,704 subtest passes/MPFR skip. These helpers have no production caller; existing scopes/host guards/downloads remain. Distillation integration and native acceptance are unfinished.

## Private retained-device distillation path

The private distillation source helper pins six forward roots with exact model/root/row/time provenance and captured arithmetic metadata. Two scalar cutoff/finite checks precede composed shape-only endpoint/primary backward; only primary gradients accumulate and coordinator objective scaling applies once. [Distillation device-input validation](pocket-tts-resident-distilldeviceinputs-20261002.md) records 36 time/logvar/scale combinations, late-row terminal retention, 4,728 independent CPU/F64 comparisons max8.97e-8 under unchanged gates, and 995 regression top-level +1,788 subtest passes/MPFR skip. No production caller, CLI identity or existing host-guard/download policy changed; source/native approval and full-step integration are unfinished.

## Opt-in retained-device integration

Separate PTX/Vulkan `resident-devicehandoff-hybrid-v1` scopes now integrate coordinator-owned row-condition children, frozen retained diagonal/distillation forward provenance and shape-only backward sources. Two scalar checks precede accumulation; forward host downloads and original overflow guards remain. [Retained-device integration validation](pocket-tts-resident-devicehandoff-20261002.md) records complete-step/update/EMA/objective/clipping/padding/token/resume parity, factory/hash identities and late publication fences, with 1,019 regression top-level +1,886 subtest passes/MPFR skip. Tiny tapes11→19/downloads47→53/elements122→132 add admission/lifetime overhead; native performance is unmeasured. Independent source/native approval and download removal are unfinished.

## Released-resource admission

The [released resource audit](pocket-tts-resident-releasedresources-20261002.md) cross-checks the actual full parameter map against independent released dimensions: 131 arrays / 89,448,706 elements. The current eight-buffer resident layout plus statistics/globals charges 2,866,551,644 bytes, above the CLI 2 GiB maximum; fake constructors reject before allocation. Removing branch/zero alone still leaves a six-buffer floor 3,478,348 bytes above 2 GiB. Released distillation forward/backward live sources charge at least 476,038,888 bytes before the transformer root. Independent scoped source review confirms the resident blocker and finds no additional coordinator lifetime/admission/scaling defect with host guards retained. Runtime ceilings/guards and native holds are unchanged; storage design and resource qualification remain unfinished.

## Remaining implementation

Broader fused device dispatch/boundary orchestration, host graph construction before pruning, inter-tape transfers/full snapshots and full-device scope are unfinished. Combined resident graph routing is implemented as a separately named hybrid with full host synchronisation each step. Attention, flow and transformer hybrids have separate replay scopes. The transformer, weighting MLP, loss reductions, clipping and AdamW/EMA run on CPU in flow mode; time embeddings and their scalar adjoints are also host work. Native numerical/resource/performance qualification and released-size training have not run.

No GPU initialisation, probe, execution, reset/recovery, model training or listening run occurred. The incident execution hold and inference listening/publication hold remain in force. The staged inference candidate stays eight paths / 469 added lines; new work is uncommitted. No complete-goal or quality/performance acceptance follows from these source and offline gates.
