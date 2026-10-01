# Native Whisper matched benchmark arm

`TestWhisperPerformanceGoalArm` prepares matched CPU/Vulkan, VAD-off/on and word-timing-off/on measurements. It is opt-in. Initial CPU/Vulkan JFK and Portuguese/French arms have executed. It does not invoke the original engine or establish performance acceptance by itself.

## Admission

Use a newly coordinated isolated process after active speech jobs drain. Match four CPU threads/affinity, memory/no-swap limits, input, language, driver and output features against the retained whisper.cpp profile. Keep production unchanged. An explicit environment flag is mandatory; ordinary tests skip trained execution.

Required variables:

| Variable | Meaning |
|---|---|
| `GO_PHERENCE_TEST_WHISPER_GOAL=1` | Explicit trained compute admission |
| `GO_PHERENCE_WHISPER_TURBO_DIR` | Pinned model/config/tokenizer/generation files used by existing Turbo gates |
| `GO_PHERENCE_WHISPER_BENCH_BACKEND` | `cpu`, `vulkan-f32`, `vulkan-f32-tile64`, `vulkan-f32-tile64-key32`, `vulkan-f32-tile64-key32-scoreilp`, `vulkan-original-q5-mlp`, `vulkan-original-q5-source`, `vulkan-q8-mlp`, or `vulkan-q8-kv-mlp` |
| `GO_PHERENCE_WHISPER_BENCH_REPEATS` | Five through ten repetitions |
| `GO_PHERENCE_WHISPER_BENCH_VAD` | Explicit `0` or `1` |
| `GO_PHERENCE_WHISPER_BENCH_WORDS` | Explicit `0` or `1` |
| `GO_PHERENCE_WHISPER_BENCH_VAD_KEEP_GAPS` | `1` explicitly retains original internal silence; absent uses compact mode. Match scheduling in comparisons. |
| `GOMEMLIMIT` | Explicit process heap budget, separately from native/container memory. Repeated grouped word alignment used `4GiB` under an `8GiB` hard cap. |
| `GO_PHERENCE_WHISPER_BENCH_LEGACY_VALUES` | `1` explicitly loads original GGML stored values widened to F32 for compatibility diagnostics; only CPU/key32 F32 modes and explicit original-Q5 MLP/source modes. Both require original Q5_0 FC1/FC2 storage; source mode also omits CPU FFN widening and returns a resident-only model. Other weights still widen to F32. |
| `GO_PHERENCE_WHISPER_GGML`, `GO_PHERENCE_WHISPER_GGML_SHA256` | Retained original GGML path/full-file pin for legacy-value mode; HF config/generation/tokenizer remain separately pinned. |
| `GO_PHERENCE_WHISPER_BENCH_INPUT` | Regular canonical mono16-kHz WAV, 10ms–60s, at most16MiB |
| `GO_PHERENCE_WHISPER_BENCH_INPUT_SHA256` | Input pin |
| `GO_PHERENCE_WHISPER_BENCH_LANGUAGE` | Explicit language policy |
| `GO_PHERENCE_WHISPER_BENCH_REPORT` | Fresh output path; existing evidence is never overwritten |
| `GOMAXPROCS=4`, `WHISPER_THREADS=4` | Matched CPU budget |
| `GO_PHERENCE_VULKAN_DEVICE` | Authorised physical device match for Vulkan arms |
| `GO_PHERENCE_SILERO_MODEL` | Pinned Silero file when VAD is enabled |

Use a bounded test timeout no longer than30minutes. Vulkan arms retain a4GiB application allocation cap and assert cleanup to the isolated native baseline. This is not a total process/RSS limit; use container/service caps externally.

The opt-in key32/gap-preserving results and limits are recorded in [bounded validation](../../docs/validation/vulkan-attention-key32-20261001.md). The exact-order score-interleaving candidate is recorded in [score ILP validation](../../docs/validation/vulkan-attention-scoreilp-20261001.md). The original-Q5 FFN mode and preparation/storage trade-off are recorded in [packed FFN validation](../../docs/validation/whisper-packed-ffn-20261001.md). The no-CPU-widening source path is recorded in [packed source validation](../../docs/validation/whisper-q5-packed-source-20261001.md). Original Q5/F16 stored-value diagnostics are recorded in [compatibility validation](../../docs/validation/whisper-original-value-bridge-20261001.md).

Diagnostic stage attribution uses `GO_PHERENCE_TEST_WHISPER_STAGES=1` with an explicit fresh `GO_PHERENCE_WHISPER_STAGES_REPORT` and timeout no longer than ten minutes. It admits baseline F32, score-ILP and `vulkan-original-q5-mlp`; the last requires pinned original GGML/HF metadata. Stage reports now include input/output names and backend to attribute projections. Individually fenced host timings differ from full graph/request time. [Original graph attribution](../../docs/validation/whisper-original-attribution-20261001.md) records the integer-dot and precision-boundary findings.

The explicit experimental `vulkan-original-q5-integer-dot` arm loads original packed-only FFN weights and negotiates integer-dot before creating the device. FFN activations/scales are Q8_1; other weights, attention and decoder remain F32. It does not reuse the bitwise-F32 contract or change defaults. [Trained candidate qualification](../../docs/validation/whisper-integer-dot-ffn-20261001.md) reports a Portuguese segment-end regression; speed/quality acceptance is not granted. Stage input capture uses an explicit `GO_PHERENCE_WHISPER_STAGE_INPUT_DUMP` directory and writes layer0 FC1/FC2 inputs exclusively without overwriting; diagnostic downloads invalidate speed interpretation of that profiling run.

The separately named `vulkan-original-q5-integer-dot-fc1` candidate quantises only FC1; FC2 keeps ordered packed-Q5/F32 arithmetic on shared original weight storage. [Timestamp attribution and FC1 qualification](../../docs/validation/whisper-integer-dot-fc1-20261001.md) records exact five-repeat small-fixture/VADword outputs and limited request gains. Independent quality/long-form/resume and matched original-speed acceptance remain open; this mode is never selected by defaults.

[Expanded FC1 qualification](../../docs/validation/whisper-fc1-expanded-quality-20261001.md) finds a podcast word/segment change without VAD, and a20ms word-boundary change with native VAD. Additional Portuguese supplied-label scoring does not establish broad accuracy or original timing parity. The candidate stays experimental. Stage profiling now admits FC1 explicitly and reports quantisation as a separate labelled stage; individually fenced samples are diagnostic, not speed acceptance.

The explicit `vulkan-original-q5-attention-unroll4` arm retains original packed FFN with F32 activations and uses the headDim64-only exact-order attention unroll. [Qualification](../../docs/validation/vulkan-attention-unroll4-20261001.md) records fresh0.9–2.0% request gains, expanded baseline-output equality and hidden-bit/cancellation gates. Other head dimensions are refused; no fallback/default selection occurs. Original timing/quality gaps remain open.

The explicit `vulkan-original-q5-decode4` arm retains original Q5 storage and ordered F32 arithmetic but assigns four lanes per block for shared decode. [Qualification](../../docs/validation/vulkan-q5-decode4-20261001.md) records native/pinned6.7–6.8% kernel gains, fresh1.3–2.2% request gains and expanded exact-output gates. It does not automatically combine attention unroll or quantisation, and defaults remain unchanged.

The explicit `vulkan-original-q5-exact-combined` arm combines the retained decode4 FFN and headDim64 F32 attention unroll kernels without quantisation. [Combined qualification](../../docs/validation/whisper-combined-exact-20261001.md) records measured2.5–3.8% fresh request gains, not summed component percentages, plus expanded exact word outputs and hidden/cancellation gates. Stage profiling admits this mode explicitly. Defaults remain unchanged; original speed/quality gaps stay open.

The explicit `vulkan-original-q5-attention-outputilp` arm keeps decode4 FFN/F32 values and interleaves four headDim64 attention output columns while preserving each key-ordered FMA chain. [Qualification](../../docs/validation/vulkan-attention-outputilp-20261001.md) records measured4.6–5.9% fresh request gains over combined exact, expanded word-output equality and hidden/cancel gates. Other head dimensions are refused; no fallback/default/quantisation change occurs. Original workflow/quality acceptance remains open.

[Exact Q5 preparation trials](../../docs/validation/vulkan-q5-preparation-20261001.md) reject signed-byte and scale-only widening: trained kernels slow down. Paired offset bytes gain only about 2% per operator while increasing packed storage by 50% and CPU packing cost. No prepared-weight mode is retained, and no request/RSS/quality gain is claimed.

## Evidence

The arm reports model loading, native/VAD preparation, cleanup and full-arm time separately from each request. Each request records allocation bytes/counts, existing decoder phase counters and full returned windows/word timing. Allocation counters include owned evidence outputs; they are not internal-only inference allocations. Word-timing decoder work contributes to phase counters when enabled.

The first request includes any lazy CPU packing/preparation. Later requests share prepared model weights but receive fresh decoder/VAD state. No hidden warm-up is discarded. Original-engine measurements must be collected separately with equivalent settings and consistent setup/load boundaries; do not compare warm Go request time with a cold original invocation and call it a speedup.

F32 and selective-Q8 arms are distinct numerical candidates. CPU versus Vulkan and VAD/timestamps on/off must be reported explicitly. Five small-fixture samples do not establish full Decoder throughput or quality. Independent VAD parity and recorded-audio timing/quality still need qualification.

The existing CPU no-VAD API and separate [experimental VAD API](pcm_vad.go) keep their respective output-coordinate contracts. Trained benchmark output is not a serving profile or checkpoint default.

Acceptance: [native Go performance contract](../../docs/speech/whisper-go-performance-contract-20260930.md). Feasibility findings: [Sigma assessment](../../docs/validation/whisper-sigma-speed-assessment-20260930.md).
