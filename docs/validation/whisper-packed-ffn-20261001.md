# Original-Q5 FFN encoder integration — 1 October 2026

Explicit packed original-Q5 FC1/FC2 weights reduce resident encoder weight arenas by 1,363,148,800 bytes and repeated request medians by about 2.4–2.6%. JFK native VAD/word requests improve by 1.8%. Preparation is more expensive; a single fresh five-request JFK pair improved full-arm time only 0.9%. The original-engine speed target is still unmet. No serving default or production binary changed.

## Owned packed weights through the graph

`NewVulkanEncoderOriginalQ5MLP` consumes a supplied pinned legacy GGML file synchronously. The supplied CPU encoder remains widened original values. For every FC1/FC2 matrix, construction verifies rank/shape and exact original Q5 decoded values against that encoder. A checkpoint with matching geometry but different MLP values fails. Other encoder weights come from the existing F32 source; callers must pair the file, encoder and decoder from the same checkpoint.

`whisperggml.File.Q5Blocks` reads owned original rank-two packed bytes in bounded chunks; only Q5_0, row-aligned matrices up to 64 MiB are admitted. Cancellation, closure and short reads return no partial data. `VkLinearQ5Set` losslessly repacks 64 FFN matrices into one aligned allocation and shares one pipeline. A two-pass layout check admits the aggregate budget before packing temporary copies. Checked stages hold private packed bindings; plans copy their metadata and retain native storage through submission. Encoder resource rollback closes plans before backing allocations and pipelines. No raw file, borrowed model byte slice or temporary block storage escapes construction.

Activations, attention, CPU decoder, bias and ordered accumulation remain F32. Original Q5 scales and quantised values are preserved without requantisation or F16 activation rounding. The explicit `vulkan-original-q5-mlp` benchmark backend requires legacy-value mode. The existing widened-value baseline and serving defaults remain unchanged.

The first benchmark helper reopened and hashed the pinned model after CPU loading. The final helper reuses the same verified open file for packed preparation, then closes it. Preparation still includes original block reads, value-identity checks, packing and native allocation/upload. CPU encoder weights remain widened in memory; these results establish GPU weight-storage savings, not elimination of CPU widening or a process-RSS reduction.

## Matched request measurements

Starting revision: `4066ef5645e65ab8aea4ba4e1df05fe9e3509dbe`. Go: `go1.26.2 linux/amd64`. Original Turbo Q5_0 model pin: `394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2`. Both arms use the same original stored values, F32 activation/decoder path, tile64 projections, score-ILP key32 online-softmax attention, four CPU threads and `GOMEMLIMIT=4GiB`. Each arm runs five requests in one isolated process; setup is separately recorded and every request is retained.

| Fixture/options | Widened FFN median (s) | Packed FFN median (s) | Request change |
|---|---:|---:|---:|
| JFK, no VAD/words | 7.095 | 6.914 | −2.55% |
| MINDS Portuguese, no VAD/words | 7.063 | 6.877 | −2.63% |
| MINDS French, no VAD/words | 6.499 | 6.346 | −2.36% |
| JFK, native VAD/words/gap preservation | 8.285 | 8.133 | −1.83% |
| Two groups, native VAD/words/gap preservation | 16.617 | 16.332 | −1.72% |
| JFK, final one-hash helper pair | 7.110 | 6.932 | −2.51% |

All five outputs per arm/fixture match returned window structures, text, tokens, segment times and, where enabled, VAD retained coordinates/eligibility and word times. The two-group fixture retained both groups without interpolation across removed audio. Request comparisons are process-local repeated work, not five independent cold launches or independent acoustic annotations.

Native encoder stats: weight arenas `2,548,039,680 → 1,184,890,880` bytes (about 53.5% less for all encoder weights). Scratch stays 93,696,000 bytes; 34 plans and 390 stages remain unchanged. Stats count logical owned storage and arena alignment, not driver padding or total RSS. Monitored process/container memory is retained separately; no total-process saving is asserted.

Preparation/full-arm trade-off is material. In the final one-hash JFK pair, preparation was `1.759 → 2.432 s`, model load `5.247 → 5.241 s`, and five-request full-arm time `42.685 → 42.307 s` (about −0.89%). Earlier double-hash arms had slower full-arm totals despite faster requests. Both sets are retained. A one-request cold-start gain is not established; frequent reload workloads may lose from the extra preparation. Full-arm and request boundaries must stay explicit.

## Correctness and recovery

The trained synthetic-mel graph test compared all 1,920,000 final hidden-state values bitwise against widened FFN execution. Baseline and candidate owners run sequentially under the same hard cap; the pinned file is closed before Forward. Early, middle and late cancellation checkpoints around 1,303 checked boundaries passed cancellation, drain where required, fresh reuse and exact hidden-state agreement. The final test ran twice during qualification, including the final budget-admission source. Native allocations returned to the starting baseline.

Offline tests cover packed-set alignment, invalid counts/shapes, raw block ownership, nonfinite scales through the existing packer, mismatched checkpoint values, nil/closed/index admission, private stage extents, plan binding retention, allocation rollback, copied source independence, cancelled in-flight plan/close and drain/reuse. Raw reads include post-final-read cancellation and truncated payloads. Deterministic selected gates passed ten repetitions.

Coverage is scoped, not repository-wide: raw reader 95.7%, identity helper 95.8%, set preparation 86.7%, set construction 76.9%, Stage 87.5%, Close 66.7%, closeLocked 77.8%, storage accessor 100%. Some construction/teardown fault branches remain unqualified; the new-set 95% target is not met. A focused independent source review timed out at 70 seconds and supplies no approval.

## Retained-tree checks and isolation

Verified: affected tests/vet; `make model-layout-check host-build host-vet host-test docs-check`; whole-tree `go test -race -p=2 -count=1 -timeout=180s ./...`; final affected-package race checks after admission hardening; and whole-tree Linux ARM64/RISC-V builds. Foreign builds are compilation evidence only. Shader modules are unchanged from the preceding verified 28-module gate; the integration changes bindings and ownership only.

The authorised Intel runner retained four-CPU quota, `taskset -c 0-7`, 8 GiB memory/memory-plus-swap cap, no network and read-only root. Qwen-idle and host available memory ≥6 GiB guards remained active. Grouped alignment kept `GOMEMLIMIT=4GiB`; the hard cap was not expanded. Final containers exited 0 without OOM or guard abort. At 07:20 UTC, native/build/candidate containers were drained, only the pre-existing `wrdp` containers remained, no Go process was active and Qwen slot 0 was idle. `@llama` received explicit release.

## Reproduction and remaining goal

[Evidence directory](../../benchmarks/speech-foundations/whisper-packed-ffn-20261001/) contains all initial/final/one-hash reports, pinned inputs/options, exact-output comparison, native recovery logs, monitors, exit states, full checks, coverage and the guarded runner. `comparison.json` includes preparation/full-arm time and native storage alongside request medians. `EncoderStats` is additive benchmark metadata; existing report fields remain intact. Report creation is exclusive; reproduce using new paths.

Native trained recovery: `GO_PHERENCE_TEST_PACKED_ENCODER=1`, pinned Turbo config/tokenizer/original-Q5 paths, named Intel device, and `TestVulkanOriginalQ5EncoderNative` with timeout no longer than 300 seconds. Trained timing uses `TestWhisperPerformanceGoalArm`, `GO_PHERENCE_WHISPER_BENCH_LEGACY_VALUES=1` and explicit `vulkan-original-q5-mlp`. No downloads or default hardware work occur in ordinary tests.

The later [streamed preparation validation](whisper-q5-stream-preparation-20261001.md) reduces temporary copies and construction time without changing the retained inference kernels or outputs. The earlier array-based measurements above remain valid for their recorded revision.

The retained original Q5 process median of 3.350 s still uses a different activation/quantisation graph and timing boundary. Go's 6.932 s request result does not meet that target. Independent acoustic/word timing, long-form/resume and matched whole-workflow acceptance remain open. Further execution-level work should remove CPU widening and packing duplication through a checked packed weight provider, or pursue measured algorithmic savings; these are proposals, not results.
