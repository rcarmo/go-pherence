# cmd/audio — speech & audio

| Command | Purpose |
|---|---|
| `whisper` | Whisper STT/translation (defaults to local large-v3-turbo weights; WAV direct and M4A/other inputs via ffmpeg; `-task`/`-language` prompt flags); supports the `WHISPER_ENC_H` EP-encoder + Go turbo-decoder hybrid |
| `diarize-vtt` | Whisper transcription/translation with optional speaker diarization → WebVTT |
| `moss-transcribe` | Native MOSS transcription, recording-local speaker diarization, and text/raw/JSON/SRT/ASS export from 16 kHz mono PCM WAV |
| `speakercheck` | Speaker-embedding / verification check |

See [`docs/moss-transcribe-diarize.md`](../../docs/moss-transcribe-diarize.md) for the MOSS support contract, real-checkpoint parity gates, usage, and limitations.

## Whisper GPU graph flags

The flags below apply to the standalone `whisper` and `diarize-vtt` commands. `moss-transcribe` selects its verified runtime-loaded NVIDIA PTX graph automatically, warns and falls back to CPU/SIMD when GPU initialisation or execution fails, and accepts `-cpu` to force the CPU oracle. Both standalone Whisper commands expose conservative GPU switches:

- `-gpu` enables the GPU-assisted encoder path when CUDA SGEMM is available and falls back to CPU/SIMD otherwise; decoder cross-K/V precompute is separately gated by `GO_PHERENCE_WHISPER_GPU_CROSS_KV=1` or `-gpu-graph`.
- `-gpu-graph` sets `GO_PHERENCE_WHISPER_GPU_GRAPH=1`, implies `-gpu`, and enables the currently wired opt-in Whisper GPU graph surfaces behind their parity/fallback guards.

Per-surface flags remain available for isolated debugging:

- `GO_PHERENCE_WHISPER_GPU_MEL=1`
- `GO_PHERENCE_WHISPER_GPU_CONV1D=1`
- `GO_PHERENCE_WHISPER_GPU_ATTENTION=1`
- `GO_PHERENCE_WHISPER_GPU_SELF_ATTN=1`
- `GO_PHERENCE_WHISPER_GPU_LM_HEAD=1`
- `GO_PHERENCE_WHISPER_GPU_CROSS_KV=1`
- `GO_PHERENCE_WHISPER_GPU_CROSS_ATTN=1`
- `GO_PHERENCE_WHISPER_GPU_DECODER_MLP=1`

Validation gates:

- `make whisper-turbo-parity` — hard JFK large-v3-turbo transcript parity; fails loudly when local assets are missing.
- `make whisper-cuda-parity` — focused numeric CPU-oracle CUDA parity gates; skips unavailable CUDA kernels on CPU-only hosts.
- `make whisper-gpu-graph-parity` — runs the transcript contract with `GO_PHERENCE_WHISPER_GPU_GRAPH=1`.
- `make whisper-turbo-check` — aggregate gate used before committing Whisper graph changes.
