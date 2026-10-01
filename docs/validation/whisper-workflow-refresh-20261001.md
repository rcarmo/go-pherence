# Whisper whole-workflow refresh — 1 October 2026

The fastest retained Go path still fails the original-speed target. On JFK, five-request medians are3.018s for the original and6.311s for Go without VAD; with genuine VAD they are3.028s and6.537s. Go's checked word alignment raises its median to7.524s. No-VAD English/Portuguese text and segment times match, but VAD output and timestamp mapping do not.

These are practical workflow measurements with pinned source/model/features/resources, not numerically equivalent core-engine comparisons. Different arithmetic, decoder placement, VAD grouping and alignment remain explicit below. No optimisation or service change is made.

## Boundary and reproducibility

- Original source: whisper.cpp `c44b60b8053bbf2a5c1e014f11323fb3f2485177`; unchanged vendored tree and existing Vulkan libraries. The diagnostic C++ executable links the existing original library, rather than calling a Go backend or reproducing its arithmetic.
- Go source: `aeddf1a604f62a75ff5c22ecf2a999089f08980c`, explicit `vulkan-original-q5-attention-outputilp`: decode4 original-Q5 FFN, F32 output-ILP flash attention, CPU checked decoder and optional CPU teacher-forced word alignment.
- Both read original turbo Q5 file SHA256 `394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2`. Genuine native Silero v6.2.0 uses SHA256 `2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987`.
- Canonical mono16kHz PCM16 WAV fixtures:11s JFK,9.045375s MINDS Portuguese row0, and55s two-copy JFK with a44s offset. Original input is the same samples converted to F32 by `int16 / 32768`, matching the checked Go reader. WAV/raw-PCM hashes and sample counts are recorded.
- Both retain a model across five requests, including the first. Go uses fresh request state; original sets `no_context=true` so previous request prompts do not accumulate. Greedy selection, explicit locale, four threads, temperature0/no temperature increment and timestamps are configured. Go additionally uses its checked model-derived generation/suppression/position policy; this is not assumed identical to original's other heuristics.
- CPU4/8GiB/no-swap, physical Intel Iris Xe, flash attention enabled, Qwen-idle and available-memory≥6GiB guards. Go uses heap target4GiB. Original logs confirm GPU/flash/Intel and DTW0. Original Silero defaults use CPU4.
- Sequential fresh processes alternate engine order across arms: original→Go for JFK noVAD, JFK timing and groups no-word; Go→original for JFK VAD no-word, PT and groups timing. Five repeats occur inside each process, not interleaved between engines or randomised.

Request walls include mel, VAD where selected, encoder, decoder and selected timestamp/alignment work. Original starts with F32 PCM already loaded; Go reads from the canonical PCM reader within requests. Both exclude model setup from request timing. Original VAD loads lazily inside its first request; Go loads VAD during preparation. First-use shaders are not prewarmed or excluded.

Cold setup and first request are reported separately from reusable requests and the full five-request arm. Go's timed setup includes model SHA checks, checked metadata/tokenizer/generation and CPU decoder loading; original's model initialization does not include external SHA verification. Therefore cold figures describe the harnesses, not isolated loader efficiency. Process/container startup and any upload/media-conversion/queue/UI/export costs are excluded; this is not an application service benchmark.

## Five-request medians

| Fixture/options | Original request (s) | Go request (s) | Go/original | Scope |
|---|---:|---:|---:|---|
| JFK, no VAD, no words | 3.018 | 6.311 | 2.09× | Same text and segment times |
| PT row0, no VAD, no words | 3.019 | 6.269 | 2.08× | Same text and segment times |
| JFK, native VAD, no words | 3.028 | 6.537 | 2.16× | VAD/window output differs |
| JFK, VAD, token/word times | 3.032 | 7.524 | 2.48× | Alignment methods differ |
| Two JFK groups, VAD, no words | 3.257 | 13.181 | 4.05× | One compact vs two gap-preserved encoder windows |
| Two JFK groups, VAD, token/word times | 3.309 | 15.157 | 4.58× | Window count and alignment differ |

Original first requests are slower than subsequent requests: for JFK noVAD4.420s versus about3.00–3.06s later. All remain included. Go's corresponding five samples range6.287–6.351s. No samples are discarded. Original's primary JFK VAD arm is rerun using the final timing-structure-cleanup harness: median3.037s, consistent with the initial3.028s, but not pooled as another matched pair.

Arithmetic is not identical: the original uses its F16/Q8_1/integer-dot paths and device offload for the decoder; Go preserves widened F32 arithmetic and keeps the decoder on CPU. The same source weights do not imply equal intermediate values or equal compute placement. These measurements establish failure of the observed workflow target, not a2× difference between equivalent encoder kernels.

## Setup and full-arm costs

| Fixture/options | Original setup+first (s) | Go load+prep+first (s) | Original five-request arm (s) | Go five-request arm (s) |
|---|---:|---:|---:|---:|
| JFK noVAD | 4.720 | 10.113 | 16.828 | 35.360 |
| PT noVAD | 4.652 | 10.310 | 16.699 | 35.405 |
| JFK VAD/no words | 4.732 | 10.366 | 16.889 | 36.530 |
| JFK VAD/timing | 4.677 | 11.378 | 16.839 | 41.449 |
| Groups VAD/no words | 5.077 | 16.963 | 18.223 | 69.683 |
| Groups VAD/timing | 5.074 | 19.067 | 18.334 | 79.720 |

Original model initialization is about0.283–0.300s. Go model loading is2.868–2.880s plus0.891–0.908s preparation. Raw reports retain exact separate setup/request/cleanup totals. Sampled container memory monitors show roughly1.0–1.04GB original and4.1–4.4GB Go; these are coarse container-accounting observations, not process RSS peaks or controlled resident-memory comparisons.

## Output and VAD semantics

Every engine repeats its own segment/text/token output exactly across its five requests. NoVAD JFK and PT text, content-token IDs and segment times match after removing original's leading whitespace; PT remains0–7.36s. No independent human/acoustic labels are used here.

With VAD, Go gap-preserving JFK produces a main segment0.322–10.382s followed by `Thank you.` at10.382–10.622s. Original produces two speech segments0.32–7.47s and8.19–10.37s without that extra text. This repeats with and without timing in every sample. The disagreement is a quality/timing failure relative to the original output, not a newly introduced regression in this unchanged Go tree, nor proof the original is an independent acoustic ground truth.

Two-group VAD compacts both speech groups into one original encoder window. Go's accepted gap-preserving mode executes two padded encoder windows. Original reports its second segment at10.37–54.38s, spanning the long removed gap; Go maps the second main speech segment to44.322–54.382s, then adds `Thank you.` after each group. Neither engine's result is accepted as an independent timestamp oracle.

A separate explicit diagnostic sets Go `PreserveWindowGaps=false`, not a default change. It uses one compacted encoder window and reduces the groups median13.181→7.469s, still slower than original3.257s. The compacted Go result splits text/times differently and does not retain the accepted word/gap semantics. Its roughly5.7s improvement attributes much of the two-group difference to encoder-window count; it is not a quality-preserving optimisation or speed acceptance. No compacted-mode word request is measured.

The original pinned initialization explicitly disables DTW token timestamps when flash attention is requested. Our timing original arm therefore uses token timestamp heuristics with DTW off; Go word timing uses CPU teacher-forcing and DTW with model-selected alignment heads. That row is not alignment-matched and cannot qualify word-timing speed/parity. A genuine original DTW/non-flash run is outside this Vulkan+flash comparison and was not silently substituted.

## Remaining cost attribution

Original arithmetic ablations use isolated fresh contexts, same JFK/noVAD/no-word settings and five requests:

| Original arm | Median request (s) |
|---|---:|
| Normal GPU/flash | 3.018 |
| Integer-dot disabled | 5.792 |
| F16 disabled, integer-dot retained | 2.897 |

These are separate diagnostic processes, not quality-accepted arithmetic baselines or independent feature deltas. Earlier full-Go quantised FFN/FC1 experiments failed expanded word/timing gates; the integer-dot advantage does not justify enabling them. The ablation shows where a materially different arithmetic path can matter, not that the accepted Go path should change precision without qualification.

The refreshed three-pass fenced Go encoder profile matches final hidden values against whole-plan execution. Mean stage costs:

| Group | Fenced mean per encoder (s) |
|---|---:|
| Attention outputILP | 1.733 |
| FC1+FC2 packed FFN | 2.029 |
| Q/K/V/O F32 projections | 1.200 |
| Conv | 0.119 |
| GELU/add/norm combined | 0.216 |

Sum≈5.297s. Stage fencing introduces synchronization; these are diagnostic costs, not additive matched request-gap estimates. The original public `whisper_get_timings` fields are **averages per call**. Decoder fields average multiple token calls, so summing them as whole-request decoder totals would be wrong. Raw original values are retained, but no additive original/Go gap decomposition uses them.

Go noVAD JFK decoder component medians are self0.171s, cross0.217s, MLP0.360s, vocabulary head0.180s. JFK VAD/no-word medians are0.203/0.258/0.424/0.209s; timing raises them to0.394/0.499/0.821/0.405s. These process-local counters also include alignment decoder work when enabled. They identify CPU decoder/MLP/head costs, not an equivalent original decoder comparison.

A separate profiled Go arm covers loading plus five noVAD JFK requests. CPU samples total15.93s over35.71s wall; `sdotAsm`39.74% and `sgemmNTTileFMA`22.35% dominate sampled CPU. Cumulative decoder `ForwardToken`3.89s and model-load path2.89s appear in the profile. Worker stacks and GPU waits mean these are not pure decoder/request wall fractions; no Amdahl speed projection is claimed. CPU profiles and raw fenced stages are archived separately from unprofiled acceptance timings.

## Verification, failures and isolation

All successful arms execute five requests, pass engine/device/feature/settings/input gates and have actual exit0/noOOM states. Native cleanup in the Go harness restores allocation counts/bytes and leaves no in-flight work. Original contexts are explicitly freed. The final C++ harness deletes the allocated public timing structures after serialization; earlier runs leaked five small timing structs, not contexts/model buffers. Both binary hashes and initial/final harness sources are preserved, and the primary VAD result is rerun on the final source.

Setup errors were retained and fixed before measurement: missing `ggml/include` in the C++ compile; failed container execution of the image's `timeout`; missing Silero path in the first Go VAD environment; and a blank source environment line converted to `=undefined` for groups. The original wrapper now uses Bun's119s process timeout plus the110s library abort callback; Go invocations use120s native test deadlines. Failed environment parsing created no container. None of these failed/empty runs enters timing results.

The evidence parser verifies five samples, per-engine repeated segment output, matching input length/locale/resources and selected feature flags. SHA256 pins bind models, inputs, binaries and the original shared library. A read-only methodology judge completed and agreed that the data support this setup's practical workflow conclusion, not an apples-to-apples core-engine or broad performance claim. It did not independently audit the raw sources or acoustic correctness.

[Hashed evidence](../../benchmarks/speech-foundations/whisper-workflow-refresh-20261001/) contains all raw output/settings/timings, startup failures, states/exits/monitors, original harness and wrapper, PCM conversion/pins, parser/comparison, CPU profile, stage profile, provenance and final retained-tree gates. No models, recordings or runtime binaries are committed. The production source is unchanged apart from this documentation/evidence. Build/test/vet/layout/docs, affected race and marked ARM64/RISC-V checks run on the retained tree before handback.

The @llama isolation window preserves Qwen LAN and Gemma state, defaults/services and resource allocations. Actual native/build/container drain precedes explicit release. The overall original-speed, independent acoustic/word timing, long-form/resume and fault acceptance remains open.

## Next work

Prioritise exact CPU decoder projection/MLP/head throughput and decoder-device placement, or a separately quality-qualified original-compatible arithmetic path. The retained GPU encoder still dominates a single window, but another minor shared-layout trial cannot explain the refreshed2× gap. VAD compaction/word-alignment contracts require independent quality gates before changing them. No goal completion, deployment or new backend is claimed.
