# Vulkan F32 64×64 projection candidate — 1 October 2026

The explicit 64×64 F32 projection candidate reduces measured Go Vulkan request latency by 18.4–19.8% on four retained English/Portuguese/French fixtures. It still takes longer than the retained original engine on JFK. Production and the default Go encoder are unchanged.

## Kernel and correctness

[Shader](../../backends/vulkan/shaders/linear_f32_regtile64.glsl): 16×16 lanes, sixteen outputs per lane, 64×64 output tile, K32 shared tile and 16 KiB shared memory. The existing 32×32 tile uses four outputs per lane. Both preserve the same ascending K reduction and F32 bias/accumulation; no quantisation, F16 activation or matrix-core path is introduced.

`NewVkLinearRegTile64F32` and `NewVulkanEncoderRegTile64` opt in explicitly. There is no automatic fallback/default switch. Admission retains shape, extent, alignment, output-alias, device-limit and cancellation checks. [Model-free tests](../../backends/vulkan/vulkan_linear_regtile64_model_test.go) shuffle group/lane ownership, check shared loads and output writers, and compare arithmetic against the existing tile and independent scalar reference, including partial tiles. Native tests cover awkward shapes and the full turbo projection dimensions with guard buffers and unchanged tolerances.

Full-shape alternating ABBA tests use eight timed dispatches per kernel. Every downloaded result equals the baseline bit-for-bit. Latest speedups over the current 32×32 kernel: 1.512× for `[1500,1280,1280]`, 1.561× for `[1500,1280,5120]`, 1.480× for `[1500,5120,1280]`. These include submission/fence wall time and exclude transfers. Native synthetic complete-encoder comparison is bit-identical to the existing encoder and passes its independent scalar tolerance; maximum absolute errors are `3.58e-7` and `1.19e-6`.

[Raw evidence and manifest](../../benchmarks/speech-foundations/vulkan-linear-regtile64-20261001/manifest.json) retain request outputs, kernel logs, independent-encoder checks and all 25 validated/rebuilt embedded shader identities.

## Request measurements

Five repetitions per arm, four CPU threads, affinity CPUs 0–7, Intel Iris Xe RPL-P/Mesa 26.1.5, isolated 8 GiB/no-process-swap container. Live Qwen idle and host available memory ≥6 GiB were guarded throughout. Models and input hashes are in every report. Go weights are loaded from the pinned 587-tensor F16 HF checkpoint into F32 inference. The existing tiled online-softmax F32 attention is unchanged.

| Fixture | Existing Go Vulkan median | Tile64 median | Reduction |
|---|---:|---:|---:|
| JFK, 11 s | 9.601 s | 7.835 s | 18.39% |
| Portuguese 0, 9.05 s | 9.549 s | 7.789 s | 18.43% |
| Portuguese 1, 4.69 s | 9.113 s | 7.361 s | 19.22% |
| French 0, 3.75 s | 9.029 s | 7.244 s | 19.77% |

All five candidate samples per fixture exactly match baseline text, token IDs, segment times and window metadata. These are request medians with VAD/word timing disabled; model load/preparation/cleanup are separately retained. Corpus runs alternate baseline/candidate by fixture, with sequential five-run arms rather than fully interleaved requests. A later unchanged JFK control median of 9.589 s corroborates the earlier 9.601 s baseline.

JFK genuine native VAD with words disabled: 9.627→7.886 s (18.08%), with all five candidate span/segment/token outputs equal. Tile64 words-on/VAD-off median: 8.895 s. VAD+word timing fails on the original checked gap guard (`word crosses a removed VAD gap; original timing unavailable`), before successful publication. No gap interpolation or timestamp tolerance change was used.

## Original engine and rejected candidates

Pinned whisper.cpp `c44b60b8053bbf2a5c1e014f11323fb3f2485177`, four threads, Intel Vulkan, flash attention, greedy beam/best-of 1, English, fallback disabled, VAD/word timing disabled. Five-run JFK process medians: 6.183 s with retained mixed F16/F32 model; 3.350 s with retained Q5_0. Original process timings include load/startup/JSON output, while Go request timings exclude load/preparation. Shader-cache coldness differs between first and later processes. Precision/storage and timing boundaries differ, so these runs locate a gap and do not establish performance acceptance. The target remains unmet even against the measured original process total.

Go CPU JFK request median: 25.873 s. Encoder stage diagnostic (single-stage submissions, three runs) spends 5.306 s in projection and 2.888 s in attention; diagnostic submission costs differ from whole-request plans. CPU profiling identifies dense NT GEMM, exact GELU and softmax as main costs.

Selective Q8 MLP request median was 11.369 s, slower than F32. K64/K128 projection tiles and shared-weight transpose were neutral or slower against the actual current 32×32 baseline. Attention K-tile transpose produced exact JFK output but no whole-request improvement (9.606 s); it was reverted. Early K-tile speedups against the older 16×16 kernel did not survive comparison against the current baseline.

## Verification and remaining work

Current-tree host build/vet/tests, full-tree race, layout/doc links, ARM64/RISC-V cross-builds and offline shader validation/rebuild pass in isolation. Native operator and complete-encoder checks pass. These gates qualify the bounded explicit candidate. They do not qualify trained long-window/cancellation, acoustic WER, natural overlap, original word-alignment parity, the full Decoder recording or deployment. Next work is attention/projection profiling, matched precision/loading boundaries and safe original-timeline VAD word alignment.

[Performance contract](../speech/whisper-go-performance-contract-20260930.md) · [Silero qualification](native-silero-parity-20261001.md).
