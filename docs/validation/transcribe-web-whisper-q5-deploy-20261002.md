# transcribe-web on the Whisper original-Q5 Vulkan path — 2 October 2026

Production `transcribe-web` on Sigma (port 8093) now runs Whisper large-v3-turbo on the whisper.cpp-compatible Vulkan path from the [final comparison](whisper-final-comparison-20261002.md). It replaces CPU Nemotron ASR. Source `2dc4a9df`, server SHA256 `83be19c3…`. The front end is unchanged (`247de771…`).

## What changed

- **Library:** `whisper.LoadOriginalQ5Model` loads the pinned GGML Q5_0 file packed-only and checks its vocabulary against the HF tokenizer. `whisper.NewVulkanEncoderOriginalQ5` builds the MMQ encoder with 48-query flash attention, tanh GELU and cross K/V, then attaches the packed Q5 rows to the AVX2 CPU decoder.
- **Stage:** `WhisperStageConfig.OriginalCompatibility` enables original decoder and window compatibility. The field is omitted when false, so existing stage identities are unchanged.
- **Resume:** with window compatibility, a resumed job replays earlier windows silently to rebuild the rolling prompt; previously it refused to resume. The test proves both that the replay happens and that the emitted output equals an uninterrupted run.
- **Server:** profile `vulkan.backend: "original-q5"`. In this mode `weights` is the GGML file and the documents are the turbo HF config, tokenizer and generation files. The multi-profile Vulkan owner cap rises from 32 to 48, the server's own limit.
- **Config:** Nemotron ASR is removed. All 36 profiles keep their IDs, languages, extensions and diarization providers. Word timestamps are back on, as in the Whisper-era configs, because speaker attribution needs word times. The config is compact JSON to stay under the 64 KiB cap.
- **Unit:** the `whisper-vulkan.conf` drop-in replaces `nemotron-cpu.conf`. Limits stay at CPUQuota 400%, MemoryMax 8G, no swap. It sets `PrivateDevices=no` and a writable Mesa cache under `data/`. Vulkan stays restricted to the Intel ICD and NVIDIA stays disabled. @llama confirmed that deployed Qwen is CPU-only and had no objection.

## Verification

| Check | Result |
|---|---|
| Gates | `go test ./...`, race (`model/whisper`, `runtime/speechjob`, `cmd/audio/speechjobserve`), arm64/riscv64 builds; `--check` passed |
| Transcript vs qualified harness (`fast6`) | JFK and PT2 cues **identical** (text and times) in candidate and production |
| Speaker labels | JFK 22/22 words labelled with Community-1 and with Nemotron (was 0 on Nemotron ASR) |
| Existing jobs | all 21 manifests byte-unchanged |
| Memory | peak 4.06 GB (limit 8G); ready in about 2–6 s |

Job wall times, including upload, queue, decode, ASR, exports and polling:

| Job | Nemotron CPU (before) | Whisper Q5 Vulkan (now) |
|---|---:|---:|
| JFK ASR (`asr-en-wav`) | — | 3.06–3.57 s (3.3 s with words) |
| JFK with speakers (`nem-en-wav`) | 7.0 s, 0 words labelled | 5.3 s, 22/22 labelled |
| JFK with Community-1 (`diar-en-wav`) | — | 5.8 s, 22/22 labelled |
| PT2, 44 s, 2 windows (`asr-pt-wav`) | — | 7.1 s |

The backup is `deploy-backup-whisper-q5-20261002T155…`: previous config, server and the `nemotron-cpu.conf` drop-in. Rollback means restoring those three files, then `daemon-reload` and a restart.
