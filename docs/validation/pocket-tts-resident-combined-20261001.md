## Pocket TTS combined resident hybrid -- 1 October 2026

`ptx-resident-hybrid` and `vulkan-resident-hybrid` now combine flow/transformer tape execution, resident gradient accumulation and resident AdamW/EMA under one runner owner. CPU handles sequence assembly, conditioner/EOS/weighting, time embeddings/scalar adjoints, interpolation/loss reductions and final norm-summary reduction. Full coherent state returns to the host each completed update. Full-device training and native qualification are unfinished.

Parent: `a90aec30567d485784f2e04d1803c543c9f2817f`. Work is uncommitted; the held inference index is eight paths / 469 insertions. No native runtime initialisation, probe, execution, reset/recovery, real model/audio or listening run occurred.

## Gradient routing

`training_resident_backend.go` binds one explicit resident owner to flow, transformer and optimiser components. Step preparation creates state lazily after restore, checks the coherent host boundary, and opens one paired step. Every graph call verifies that all three components still share that owner. It builds tapes using current host shape metadata, binds resident parameters read-only and downloads only boundary output/input/time adjoints. Device parameter gradients stay in resident accumulation buffers; host arrays for those components remain zero.

The diagonal objective explicitly accumulates its seeded backward once. Minimal-LSD primary and endpoint forward passes and the endpoint VJP all disable parameter accumulation. Only the primary mixed backward accumulates, scaled by `(1-p_equal)/valid_rows`; endpoint input derivatives still return upstream. Transformer forward disables accumulation and its loss-seeded re-evaluation accumulates once. Objective weight and valid-row scaling are kept distinct from host conditioning/target derivative scaling.

The complete tiny step executes two transformer tapes and twelve flow tapes for two valid rows. Tests check exactly four accumulations per flow parameter and one per transformer parameter, and exactly 100 boundary tensor downloads. No parameter-gradient tensors are downloaded. All accumulated gradients at optimiser admission match the independent pinned PyTorch full-step fixture within unchanged 5e-5 bounds. Host tests cover p_equal 0/.25/1, a separate EOS weight, masked padding and preservation of the input-only endpoint boundary.

CPU-owned conditioner/EOS/weighting gradients upload once after graph derivatives. Device-owned host gradient arrays must stay zero; a nonzero value rejects double accounting. The resident coordinator requires every named parameter gradient before global clipping/update. Post-update summaries and full snapshot validation precede host LoadState, then runner replay commits. Failure after partial device work leaves host parameters/moments/EMA and replay at the prior boundary, poisons the runner and blocks checkpoint/retry. Active paired steps cannot publish or close normally.

## Replay and resources

Separate `*-resident-hybrid-v1` replay scopes bind the runtime digest. Construction/update/checkpoint/restore reject mixed or replaced component owners. Previous SIMD, attention, flow, transformer and optimiser v1 option identities are unchanged. Combined scope requires sampler/native opt-in and a runtime pin; full `ptx`/`vulkan` stays rejected.

The CLI byte ceiling separately caps resident logical state and one live tape. Their combined logical peak can approach twice that value, before native metadata/alignment and host graph/snapshot memory. This baseline still copies host parameter data while constructing graphs, allocates intermediates per invocation and re-evaluates transformer forward with the loss seed. Parameters persist on device between updates, but a complete host snapshot synchronises every step. No latency, allocation, peak-RSS or released-size resource acceptance exists.

## Offline verification

Pinned fixture: `training_step_pytorch.json`, upstream `0acce6b2f390150267557770d2098c5caa9a18ac`. Combined host-tape step matches loss at 3e-5, input/all parameter gradients at 5e-5 and updated parameter/EMA at 4e-6, all unchanged. Update/resume tests compare moments/EMA and raw replay continuation, and verify resident allocation counts do not grow across steps. Host mocks replace native execution; none of these checks measures device arithmetic.

Failure tests cover transformer admission, primary/endpoint/mixed flow stages, optimiser failures after partial parameter updates, host mutation, wrong model owners, missing/changed components, bad gradient topology, invalid ceilings and active boundary guards. A failed paired step cannot retry or checkpoint. The independent read-only review timed out after 60 seconds without findings or approval.

Final main gate: **23:08:50.913486430--23:09:28.106604550 UTC**, 1 October 2026, exit 0. Nine selected tests repeated ten times gave **90 top-level and 20 subtest passes**; race gave nine top-level and two subtest passes. Affected regressions gave **68 top-level and 80 subtest passes**. Added owner/topology guards then passed ten times (ten top-level / 60 subtests) plus race in **23:10:29.141837790--23:10:50.080814914 UTC**. No final failures or skips.

Controls: local Go 1.26.3, `GOENV=off`, empty `GOFLAGS`, `GOPROXY=off`, `GOSUMDB=off`, `GOMAXPROCS=2`, both GPU disable flags, nice10, serial packages/tests, 30-second inner / 120-second outer deadlines. Vet, whole-tree host build, Linux ARM64/RISC-V CLI cross-builds, layout (Go docs tests, five Bun tests / 3,008 assertions) and whitespace checks pass. Foreign binaries were not executed. No kernel source changed in this slice.

Final scoped coverage: identity/admission-owner guards 100%, PrepareTrainingStep 82.1%, flow evaluation 80%, transformer evaluation 100%, UpdateTrainer 75%, boundary validation 90%, Close 72.7%. Flow boundary decoding is 91.3%; selected package total is about 34%. Native transfer/dispatch/cleanup failure branches and larger model/resource/numerical ranges are open. No independent approval was obtained.

Archive: `/workspace/exports/pocket-gpu-training-resident-combined-20261001/`. Remaining implementation includes device conditioner/EOS/weighting/loss/time-embedding orchestration and all-device global norm, plus eliminating per-tape graph/activation churn. Native numerical/performance/resource, statistical sampler and new-voice/listening acceptance require authorised work.
