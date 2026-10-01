# Native Whisper matched benchmark arm

`TestWhisperPerformanceGoalArm` prepares matched CPU/Vulkan, VAD-off/on and word-timing-off/on measurements. It is opt-in. Initial CPU/Vulkan JFK and Portuguese/French arms have executed. It does not invoke the original engine or establish performance acceptance by itself.

## Admission

Use a newly coordinated isolated process after active speech jobs drain. Match four CPU threads/affinity, memory/no-swap limits, input, language, driver and output features against the retained whisper.cpp profile. Keep production unchanged. An explicit environment flag is mandatory; ordinary tests skip trained execution.

Required variables:

| Variable | Meaning |
|---|---|
| `GO_PHERENCE_TEST_WHISPER_GOAL=1` | Explicit trained compute admission |
| `GO_PHERENCE_WHISPER_TURBO_DIR` | Pinned model/config/tokenizer/generation files used by existing Turbo gates |
| `GO_PHERENCE_WHISPER_BENCH_BACKEND` | `cpu`, `vulkan-f32`, `vulkan-f32-tile64`, `vulkan-f32-tile64-key32`, `vulkan-f32-tile64-key32-scoreilp`, `vulkan-original-q5-mlp`, `vulkan-q8-mlp`, or `vulkan-q8-kv-mlp` |
| `GO_PHERENCE_WHISPER_BENCH_REPEATS` | Five through ten repetitions |
| `GO_PHERENCE_WHISPER_BENCH_VAD` | Explicit `0` or `1` |
| `GO_PHERENCE_WHISPER_BENCH_WORDS` | Explicit `0` or `1` |
| `GO_PHERENCE_WHISPER_BENCH_VAD_KEEP_GAPS` | `1` explicitly retains original internal silence; absent uses compact mode. Match scheduling in comparisons. |
| `GOMEMLIMIT` | Explicit process heap budget, separately from native/container memory. Repeated grouped word alignment used `4GiB` under an `8GiB` hard cap. |
| `GO_PHERENCE_WHISPER_BENCH_LEGACY_VALUES` | `1` explicitly loads original GGML stored values widened to F32 for compatibility diagnostics; only CPU/key32 F32 modes and explicit original-Q5 MLP mode. The latter requires original Q5_0 FC1/FC2 storage; other weights still widen to F32. |
| `GO_PHERENCE_WHISPER_GGML`, `GO_PHERENCE_WHISPER_GGML_SHA256` | Retained original GGML path/full-file pin for legacy-value mode; HF config/generation/tokenizer remain separately pinned. |
| `GO_PHERENCE_WHISPER_BENCH_INPUT` | Regular canonical mono16-kHz WAV, 10ms–60s, at most16MiB |
| `GO_PHERENCE_WHISPER_BENCH_INPUT_SHA256` | Input pin |
| `GO_PHERENCE_WHISPER_BENCH_LANGUAGE` | Explicit language policy |
| `GO_PHERENCE_WHISPER_BENCH_REPORT` | Fresh output path; existing evidence is never overwritten |
| `GOMAXPROCS=4`, `WHISPER_THREADS=4` | Matched CPU budget |
| `GO_PHERENCE_VULKAN_DEVICE` | Authorised physical device match for Vulkan arms |
| `GO_PHERENCE_SILERO_MODEL` | Pinned Silero file when VAD is enabled |

Use a bounded test timeout no longer than30minutes. Vulkan arms retain a4GiB application allocation cap and assert cleanup to the isolated native baseline. This is not a total process/RSS limit; use container/service caps externally.

The opt-in key32/gap-preserving results and limits are recorded in [bounded validation](../../docs/validation/vulkan-attention-key32-20261001.md). The exact-order score-interleaving candidate is recorded in [score ILP validation](../../docs/validation/vulkan-attention-scoreilp-20261001.md). The original-Q5 FFN mode and preparation/storage trade-off are recorded in [packed FFN validation](../../docs/validation/whisper-packed-ffn-20261001.md). Original Q5/F16 stored-value diagnostics are recorded in [compatibility validation](../../docs/validation/whisper-original-value-bridge-20261001.md).

## Evidence

The arm reports model loading, native/VAD preparation, cleanup and full-arm time separately from each request. Each request records allocation bytes/counts, existing decoder phase counters and full returned windows/word timing. Allocation counters include owned evidence outputs; they are not internal-only inference allocations. Word-timing decoder work contributes to phase counters when enabled.

The first request includes any lazy CPU packing/preparation. Later requests share prepared model weights but receive fresh decoder/VAD state. No hidden warm-up is discarded. Original-engine measurements must be collected separately with equivalent settings and consistent setup/load boundaries; do not compare warm Go request time with a cold original invocation and call it a speedup.

F32 and selective-Q8 arms are distinct numerical candidates. CPU versus Vulkan and VAD/timestamps on/off must be reported explicitly. Five small-fixture samples do not establish full Decoder throughput or quality. Independent VAD parity and recorded-audio timing/quality still need qualification.

The existing CPU no-VAD API and separate [experimental VAD API](pcm_vad.go) keep their respective output-coordinate contracts. Trained benchmark output is not a serving profile or checkpoint default.

Acceptance: [native Go performance contract](../../docs/speech/whisper-go-performance-contract-20260930.md). Feasibility findings: [Sigma assessment](../../docs/validation/whisper-sigma-speed-assessment-20260930.md).
