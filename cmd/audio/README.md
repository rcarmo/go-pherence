# cmd/audio — speech & audio

| Command | Purpose |
|---|---|
| `whisper` | Whisper STT/translation (defaults to local large-v3-turbo weights; WAV direct and M4A/other inputs via ffmpeg; `-task`/`-language` prompt flags); supports the `WHISPER_ENC_H` EP-encoder + Go turbo-decoder hybrid |
| `diarize-vtt` | Whisper transcription/translation with optional speaker diarization → WebVTT |
| `moss-transcribe` | Native MOSS transcription, recording-local speaker diarization, and text/raw/JSON/SRT/ASS export from 16 kHz mono PCM WAV |
| `nemotron` | Pinned Nemotron ASR text or diarization JSON segments from mono 16-kHz WAV; SIMD, explicit PTX/Vulkan projection hybrid, or opt-in resident Vulkan diarization tower hybrid |
| `speakercheck` | Speaker-embedding / verification check |
| [`speechjob`](speechjob/README.md) | Authenticated HTTP job client: upload/status/run/cancel/delete, explicit queue commands and verified no-clobber transcript downloads; no inference server/model loading |
| [`speechjobserve`](speechjobserve/README.md) | Opt-in Linux/amd64 server with one local hash-pinned checked CPU Whisper/FFmpeg ASR profile, metadata check mode, opt-in durable queue/worker and owned HTTP shutdown; not deployed or trained-qualified |

See [`docs/speech/moss-transcribe-diarize.md`](../../docs/speech/moss-transcribe-diarize.md) for the MOSS support contract, real-checkpoint parity gates, usage, and limitations.

`nemotron` uses the released checkpoints and processes PCM in five-second calls:

```sh
go run ./cmd/audio/nemotron -task asr -backend simd \
  -input testdata/jfk.wav -model checkpoints/nemotron/asr/model.safetensors
go run ./cmd/audio/nemotron -task diarization -backend simd \
  -input testdata/jfk.wav -model checkpoints/nemotron/diarization/model.safetensors
```

ASR prints decoded text; diarization prints JSON speaker spans. `-backend ptx`
or `-backend vulkan` normally offloads only the ASR subsampling projection
or the diarization stacking projection. The encoder/tower and prediction/head
stay on CPU in that mode. An experimental `-task diarization -backend vulkan
-vulkan-tower` also executes the 30 later audio layers and final normalisation
on Vulkan with intermediate activations resident. The frontend, first audio
layer, speaker cache and head remain on CPU. GPU errors fail the request
without a CPU fallback. On an i7-12700/RTX 3060 with `GOMAXPROCS=4`, this
opt-in path took 9.45 s on an 11-second JFK WAV versus 2.03 s on SIMD;
it is a parity path, not a request-level speedup. The reported timer excludes
WAV/checkpoint loading but includes GPU preparation, transfers and teardown. Input must be mono 16-kHz WAV; ASR
uses `tokenizer.json` beside the checkpoint unless `-tokenizer` is set.
See the [ASR](../../docs/speech/nemotron-asr-roadmap.md) and
[diarization](../../docs/speech/nemotron-3-diarization-roadmap.md)
roadmaps for pinned parity results and unresolved accuracy/GPU gates.

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
