# Whisper large-v3-turbo execution graph coverage

This note tracks the go-pherence Whisper execution graph against the whisper.cpp-style pipeline, focusing on `openai/whisper-large-v3-turbo` as the default `cmd/audio/diarize-vtt` model.

## Default model and prompt contract

| Item | Current default |
|---|---|
| Model | `openai/whisper-large-v3-turbo` |
| Local weights | `models/whisper-large-v3-turbo-hf/model.safetensors` |
| Tokenizer | `models/whisper-large-v3-turbo-hf/tokenizer.json` |
| Command | `cmd/audio/diarize-vtt` |
| Size flag | `-size turbo` |
| Task/language | `-task translate -language en` |

Turbo parity note: on a Portuguese voice-memo clip, both Transformers and Go produce English translation with `language=en`. `language=pt`/`portuguese` can transcribe/repeat Portuguese under turbo, so the default prompt is intentionally `en` for translated VTT.

## Graph stages

| Stage | Shape / role | Default backend | Optional / experimental backend | Fidelity status |
|---|---|---|---|---|
| Audio decode | arbitrary input → 16 kHz mono WAV samples | `ffmpeg` materialization in audio commands, direct WAV loader for WAV | n/a | Functional; `speakercheck` also falls back to ffmpeg for non-WAV. |
| Mel / Fbank | Whisper log-mel `[128,T]` | `loader/audio.MelSpectrogram` now attempts the amd64 fused SIMD mel path first, with scalar/SIMD FFT fallback elsewhere; fused kernels use Whisper-compatible `log10` | CUDA/PTX fused mel body exists as a correctness-first direct-DFT kernel under `backends/cuda/ptx/fft.go` and is opt-in through `GO_PHERENCE_WHISPER_GPU_MEL=1` or umbrella `GO_PHERENCE_WHISPER_GPU_GRAPH=1` in the shared Whisper mel helper | CPU/SIMD fused path is wired into production mel preprocessing and has a log-base/reference regression test; CUDA/PTX mel has a callable graph body but still needs optimized FFT dispatch before default use. |
| Encoder conv stem | mel → encoder frames | Go/RVV/SGEMM optimized conv/im2col paths in `models/whisper/encoder.go` and SIMD backends | `GPUEncoder` can opt into correctness-first CUDA/PTX Conv1D stem dispatch with `GO_PHERENCE_WHISPER_GPU_CONV1D=1`; K3/RVV conv reformulation for native builds | Covered in CPU/SIMD and GPU-assisted paths; CUDA Conv1D dispatch is opt-in until whole-pipeline measurements justify default use. |
| Encoder attention | self-attention per encoder layer | Native Go/SIMD, int8 IME when `WHISPER_INT8=1` | FP16 RVV attention with `WHISPER_FP16_ATTN=1`, head batching with `WHISPER_FP16_HEAD_BATCH=1`; correctness-first CUDA/PTX encoder attention dispatch via `GO_PHERENCE_WHISPER_GPU_ATTENTION=1` | Int8 path validated byte-identical on fixtures; FP16 paths are transcript-safe but not default because they are slower; CUDA attention dispatch is opt-in until a tiled kernel proves whole-pipeline benefit. |
| Encoder FFN FC1/FC2 | `1280→5120→1280` | Native int8 IME/SIMD path when `WHISPER_INT8=1`; otherwise checked native SIMD `SgemmNTTo` tile path with scalar fallback | A100 FFN family: `WHISPER_A100_FC1`, `WHISPER_A100_FC2`, `WHISPER_A100_FFN_FUSED`, `WHISPER_A100_X100_PACK`, `WHISPER_A100_NATIVE_Q8`, `WHISPER_A100_FFN_FC2_MODE`, `WHISPER_A100_FFN_LAYERS`, `WHISPER_FFN_TILE_M` | Row-scale A100 fused FFN is fastest and token-preserving on `pod_30.wav`, but remains opt-in pending broader validation. Native SIMD SGEMM is the oracle/default fallback; `cmd/audio/whisperffndiag` compares A100 FFN variants against it. |
| Cross-K/V precompute | encoder output → decoder cross-K/V | Go/SIMD precompute by default | GPU SGEMM cross-K/V precompute with `GO_PHERENCE_WHISPER_GPU_CROSS_KV=1` or umbrella `GO_PHERENCE_WHISPER_GPU_GRAPH=1`; GPU-resident K/V upload remains tied to `GO_PHERENCE_WHISPER_GPU_CROSS_ATTN=1` | Precompute covered; per-token GPU cross-attention regressed in earlier tests and remains separately opt-in. |
| Decoder self-attention | autoregressive causal cache | Go/SIMD / int8 decode path | `GO_PHERENCE_WHISPER_GPU_SELF_ATTN=1` can route the single-query cached self-attention through the correctness-first CUDA/PTX attention wrapper; TurboQuant self-KV cache exists but is not useful for short VAD-packed chunks | CPU/SIMD remains the default oracle; GPU self-attn is opt-in because per-token cache upload/launch overhead is expected to dominate until residency is improved. |
| Decoder cross-attention | token query attends to encoder K/V | Go/SIMD | GPU cross-attn opt-in `GO_PHERENCE_WHISPER_GPU_CROSS_ATTN=1`, now dispatching through the Whisper-specific CUDA/PTX `WhisperCrossAttentionBuffer` wrapper | CPU/SIMD path is default; GPU per-token launch/transfer overhead still blocks default. |
| Decoder MLP | 4 turbo decoder layers | Go/SIMD / int8 decode path | Resident GPU MLP opt-in `GO_PHERENCE_WHISPER_GPU_DECODER_MLP=1` | Covered; GPU MLP regressed on current workload because per-token launch overhead dominates. |
| LM head | decoder state → vocab logits | Go/SIMD, optional int8 | NVIDIA LM-head kernel when uploaded | Covered. |
| Decode policy | greedy no-timestamps, repetition guards, dynamic token budget | `GreedyDecodePrompt`, VAD-packed chunk loop in `diarize-vtt` | Speculative scaffolding exists (`ForwardTokens`, `SpeculativeDecodePrompt`) | Greedy path is production; speculative verifier remains sequential/correctness-only. |
| Timestamp/cue assembly | chunks → WebVTT cues | VAD-packed jobs, overlap-aware cue timing, progressive write/resume | n/a | Covered; resume skips by interval coverage and cleans stale cues. |
| Speaker diarization | VAD → ECAPA embeddings → clustering | SpeechBrain ECAPA Go port with parity-checked Fbank, singleton smoothing | `speakercheck` validation harness, JSON/expected label scoring | Speaker model remains opt-in for `diarize-vtt`; broader labeled multi-speaker suite still needed before defaulting. |

## Inference modalities currently supported

| Modality | Entry point | Status |
|---|---|---|
| Single audio transcription | `cmd/audio/whisper`, `Whisper.TranscribePrompt`, `Whisper.TranscribeFromSamplesPrompt`, and `Whisper.TranscribeWithLanguageDetect` | Native/optimized STT/translation path; large-v3/turbo supported by `-size`; standalone command and file/sample APIs expose explicit task/language prompts; standalone `-gpu` opts into GPU-assisted encoder/LM-head paths with CPU/SIMD fallback, while `-gpu-graph` enables the full currently wired opt-in graph behind parity gates; A100 comparison also covers timestamp translate/transcribe; timestamp mode suppresses prompt/control specials while allowing timestamp tokens and applies text repetition guards; language-detect transcription uses the shared mel path and standard prompt decode helper. |
| Long-form translated VTT | `cmd/audio/diarize-vtt` | Default user-facing path; now defaults to turbo + English translation prompt; JFK smokes cover default translate, explicit transcribe, speaker-tagged translate, and speaker-tagged transcribe VTT; `-gpu=false -workers 1 -max-tokens 16` emits one translated cue at RTF≈0.98; `-gpu-graph` is smoke/parity-gated and falls back on CPU-only hosts. |
| Speaker-only validation | `cmd/audio/speakercheck` | VAD/ECAPA/clustering without Whisper; supports WAV and ffmpeg-readable inputs, JSON, and `-expect`. |
| Batched encoder | `Encoder.BatchedForward` and `BatchedChunkedTranscribe` | Present for multiple chunks; batched chunk mel extraction now uses `loader/audio.MelSpectrogram` instead of placeholder zero features. |
| Chunked/streaming transcription | `models/whisper/chunked.go`, `models/whisper/batched.go`, `cmd/audio/whisper`, `cmd/audio/diarize-vtt` | VAD-packed chunks, overlap, progressive write, resume; all chunk/language-detect mel paths now route through `loader/audio.MelSpectrogram` instead of placeholder features; standalone no-timestamp mode chunks long inputs with no-timestamp per-window decode instead of over-length encoder passes and is covered by `whisper_turbo_smoke`; standalone timestamp output filters empty/punctuation-only cues. |
| Speculative decode | `models/whisper/speculative*.go` | Correctness scaffold only; target verifier state now rolls back rejected draft KV and replays only accepted tokens plus bonus; no speedup until verifier batching or smaller drafter is integrated. |

## Native SIMD oracle policy

For large-v3-turbo backend work, the native CPU/SIMD path is the correctness oracle for refining NEON, RISC-V/IME, A100, NVIDIA, and CUDA/PTX variants. New hardware kernels should either compare directly against the checked SIMD/scalar facade on synthetic fixtures or preserve transcript/VTT parity through `make whisper-turbo-check`, `make whisper-backend-compare`, `make whisper-backend-podcast-compare`, and windowed `scripts/whisper_a100_compare.py` runs before being considered for default use.

## Custom kernel coverage

| Kernel family | Location | Default? | Notes |
|---|---|---|---|
| RVV/SIMD SGEMM, dot, norm | `backends/simd/runtime`, `backends/spacemit/rvv` | Yes where available | Core CPU path and oracle for encoder/decoder; Whisper linear tiles route through checked `SgemmNTTo` with scalar fallback, and RVV asm wrappers are riscv64-only with scalar non-riscv fallbacks so tests build on development hosts. |
| IME int8 matmul | `backends/spacemit/ime2`, `backends/spacemit/k3engine/aipool` | Yes for K3 when `WHISPER_INT8=1`; command defaults vary by build/runtime | Native K3 high-throughput path; validate parity against the native SIMD oracle with `make whisper-int8-compare`; non-K3 builds keep safe disabled stubs for validation. |
| A100 Q8 FFN | `models/whisper/a100_int8.go`, `backends/spacemit/k3engine/aipool` | Opt-in | Row-scale native-Q8 mode is default-candidate, not automatic. |
| NVIDIA SGEMM / LM-head / GPU encoder | `backends/nvidia`, `models/whisper/gpu_encoder.go` | GPU-assisted when requested/available | Full decoder GPU residency remains not default; unused placeholder-only GPU transpose hooks were removed so this row reflects real GPU-assisted surfaces only. |
| CUDA/PTX mel/conv/attention/pool scaffolding | `backends/cuda/ptx` plus NVIDIA mega-module entries/wrappers | No | Correctness-first mel, Conv1D k=3 stride 1/2, attention/cross-attention, and attentive-stat-pool PTX bodies now exist, are registered as loadable NVIDIA runtime entries, have Buffer launch wrappers, and contract-test load/fma/softmax/sqrt/log/store structure; mel, Conv1D, encoder attention, decoder self-attention, cross-K/V precompute, decoder cross-attention, and decoder MLP have opt-in model dispatch through umbrella `GO_PHERENCE_WHISPER_GPU_GRAPH=1` or per-surface flags (`GO_PHERENCE_WHISPER_GPU_MEL=1`, `GO_PHERENCE_WHISPER_GPU_CONV1D=1`, `GO_PHERENCE_WHISPER_GPU_ATTENTION=1`, `GO_PHERENCE_WHISPER_GPU_SELF_ATTN=1`, `GO_PHERENCE_WHISPER_GPU_CROSS_KV=1`, `GO_PHERENCE_WHISPER_GPU_CROSS_ATTN=1`, `GO_PHERENCE_WHISPER_GPU_DECODER_MLP=1`), while optimized resident decoder dispatch remains pending. |
| SpeechBrain ECAPA | `models/speaker` | Opt-in speaker model | Fbank and ECAPA parity validated; broader labeled quality suite pending. |

## Default eligibility snapshot

| Candidate | Decision |
|---|---|
| `large-v3-turbo` VTT model | **Default** with `-language en`; CPU/SIMD JFK transcript parity is locked by `TestLargeV3TurboJFKCPUTranscriptParity`. |
| Full `large-v3` | Reference/fallback for source-language prompt behavior and quality comparisons. |
| K3 native int8 | Safe/validated optimized path for K3 native runs; `make whisper-int8-compare` checks stdout, standalone timestamp/VTT, diarize-vtt VTT, and speaker-tagged diarize-vtt VTT parity for translate/transcribe against the native SIMD oracle. |
| A100 row-scale fused FFN | Strong opt-in/default candidate; validate with `make whisper-backend-compare` / `make whisper-a100-compare` / `scripts/whisper_a100_compare.py` across standalone stdout, timestamp VTT, diarize-vtt, and speaker-tagged diarize-vtt outputs for translate/transcribe before automatic default. |
| GPU decoder MLP / cross-attn | Keep opt-in; prior measurements regressed due to launch/transfer overhead. |
| Whisper speculative decode | Keep correctness-only until verifier batching/drafter integration. |
| ECAPA speaker labels | Keep opt-in until broader labeled multi-speaker validation passes. |

## Remaining high-impact gaps

1. Build a broader labeled validation suite for turbo VTT + speaker labels, including the voice-memo M4A windows.
2. Validate A100 row-scale fused FFN and K3 int8/IME on multiple long-form clips with `make whisper-backend-compare`, `make whisper-backend-podcast-compare`, or repeated `--audio` plus optional `--start/--duration` arguments to `scripts/whisper_a100_compare.py`. Current non-riscv smoke coverage includes JFK and a 12s podcast window at 300s; stdout (`This one here is`), timestamp/VTT (`This one here`), diarize-vtt, and speaker-tagged diarize-vtt podcast comparisons match baseline.
3. Decide whether `WHISPER_A100_FFN_FUSED=1 WHISPER_A100_X100_PACK=1 WHISPER_A100_NATIVE_Q8=1` can become default on K3/A100-capable hosts.
4. Implement or reject fused GPU mel and resident decoder kernels based on whole-pipeline measurements, not isolated kernel timings.
5. Turn speculative decode from sequential correctness scaffold into a batched verifier path if a drafter/reference split becomes available.
