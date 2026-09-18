# Speech commands

[Command index](commands.md) | [Runtime tuning](tuning.md)

## `moss-transcribe` — native transcription and diarization

`cmd/audio/moss-transcribe` executes the pinned MOSS-Transcribe-Diarize checkpoint entirely in Go: Whisper-Medium audio encoding, temporal merge/VQ adaptor, Qwen3 greedy decoding, recording-local speaker labels, and timestamps. The verified runtime-loaded NVIDIA PTX graph is selected automatically; unavailable stages fall back to CPU/SIMD with a warning, and `-cpu` forces the CPU oracle. Whisper encoder weights and hot buffers remain GPU-resident, the VQ adaptor runs on GPU, and Qwen3 uses arbitrary-embedding batched prefill plus resident KV, decode, and LM-head state. The binary remains zero-CGo and does not require a CUDA toolkit or native SDK.

```bash
make moss-transcribe
bin/moss-transcribe -capabilities

bin/moss-transcribe \
  -model-dir /path/to/MOSS-Transcribe-Diarize \
  -audio meeting.wav \
  -format json \
  -output meeting.json
```

Output formats are `text`, `raw`, `json`, `srt`, and `ass`. Input is native 16 kHz mono PCM WAV; arbitrary media containers are not decoded. Generation is greedy only and capped at 5,120 new tokens. Sampling controls, context overflow, and unsupported checkpoint revisions fail explicitly. Speaker labels such as `S01` identify speakers only within the recording, not across recordings.

Run all pinned synthetic and real-checkpoint parity gates with:

```bash
make moss-transcribe-parity \
  MOSS_TRANSCRIBE_MODEL_DIR=/path/to/MOSS-Transcribe-Diarize
```

See [moss-transcribe-diarize.md](../speech/moss-transcribe-diarize.md) for pinned revisions, the native graph contract, exact JFK transcript parity, internal numerical tolerances, and measured CPU/SIMD and RTX 3060 performance.


## `whisper` — standalone turbo STT/translation

`cmd/audio/whisper` is the simpler single-command Whisper entry point. It defaults to local `openai/whisper-large-v3-turbo` weights and accepts WAV directly or other audio formats through `ffmpeg`.

```bash
go run ./cmd/audio/whisper \
  -audio meeting.m4a \
  -task translate \
  -language en
```

Useful flags:

- `-model PATH` — default `models/whisper-large-v3-turbo-hf/model.safetensors`.
- `-size turbo|large-v3|...` — default `turbo`.
- `-task transcribe|translate` — default `transcribe`; use `translate` for English translation.
- `-language CODE` — default `en`; for turbo translated English, keep `en`.
- `-timestamps` — emit timestamped segments; with `-output something.vtt`, writes WebVTT.
- `-diarize` — add speaker labels when `-timestamps` is enabled and the speaker model is available.
- `-chunk N -chunk-workers N` — long-form windowing controls; simple no-timestamp mode also chunks long inputs instead of sending over-length audio to the encoder.
- `-max-tokens N` — cap decoder output for smokes/benchmarks.
- `-gpu` — opt into the GPU-assisted encoder/LM-head path when CUDA SGEMM is available; falls back to CPU/SIMD otherwise.
- `-gpu-graph` — enable `GO_PHERENCE_WHISPER_GPU_GRAPH=1` for the full opt-in Whisper GPU graph surface; implies `-gpu` and remains parity/fallback guarded.

Quick turbo smoke:

```bash
go run ./cmd/audio/whisper -audio testdata/jfk.wav -task translate -language en -max-tokens 4
```


## `diarize-vtt` — turbo translated WebVTT

`cmd/audio/diarize-vtt` is the current long-form audio command. It now defaults to Whisper large-v3-turbo translation, VAD-packed chunks, progressive writes, and resume support:

```bash
go run ./cmd/audio/diarize-vtt \
  -input meeting.m4a \
  -output meeting.vtt
```

Useful flags:

- `-task translate|transcribe` — default `translate`.
- `-language CODE` — Whisper language prompt (`en` default for turbo English translation; use `pt`/`es` for source-language prompts or full large-v3 behavior).
- `-workers N` — default `min(16, runtime.NumCPU())`; local stress testing found 16 best and 20 regressed.
- `-chunk 10 -overlap 1 -vad-pack=true` — default VAD-packed chunk profile.
- `-max-tokens 40 -tokens-per-sec 4` — tuned decoder token budget.
- `-progressive=true -resume=true` — preserve and resume partial VTTs.
- `-gpu=true` — GPU-assisted encoder and LM head when CUDA SGEMM is available; cross-K/V precompute remains separately gated by `-gpu-graph` or `GO_PHERENCE_WHISPER_GPU_CROSS_KV=1`.
- `-gpu-graph=true` — enable `GO_PHERENCE_WHISPER_GPU_GRAPH=1` for all currently wired opt-in Whisper GPU graph surfaces; implies `-gpu=true`.
- `-speaker-model PATH` — optional converted ECAPA safetensors speaker embedding model.
- `-speaker-threshold 0.3` — cosine similarity threshold for speaker clustering.

Current limitations: speaker labels remain a single-speaker fallback unless `-speaker-model` points to converted ECAPA weights; `GO_PHERENCE_WHISPER_GPU_GRAPH=1`, `GO_PHERENCE_WHISPER_GPU_DECODER_MLP=1`, and `GO_PHERENCE_WHISPER_GPU_CROSS_ATTN=1` are experimental and slower on the current stress sample. See [whisper-diarize-vtt.md](../speech/whisper-diarize-vtt.md).


## `speakercheck` — speaker-only ECAPA validation

Use this to validate VAD → ECAPA embeddings → clustering without loading Whisper. WAV files are read directly; other audio formats such as M4A are decoded through `ffmpeg` when available.

```bash
go run ./cmd/audio/speakercheck \
  -input testdata/jfk.wav \
  -speaker-model models/speaker-ecapa-voxceleb.safetensors \
  -threshold 0.3 \
  -context 0.5
```

For long recordings, spot-check a short window:

```bash
go run ./cmd/audio/speakercheck \
  -input testdata/podcast.wav \
  -speaker-model models/speaker-ecapa-voxceleb.safetensors \
  -start 300 \
  -duration 30 \
  -sims=false
```

It prints VAD segment timings, assigned speaker labels, speaker counts, and optional pairwise cosine similarities. Add `-json` to emit a machine-readable report. Add `-expect 1,1,2,2` to score a labeled fixture by exact label accuracy and pairwise same/different agreement; text mode exits non-zero when pairwise score is below 1.

Run a repeatable labeled suite with:

```bash
python3 scripts/speakercheck_suite.py testdata/speakercheck_suite.json
```
