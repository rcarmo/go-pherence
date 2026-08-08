# go-pherence

![go-pherence](docs/icon-256.png)

go-pherence is a Go inference toolkit for running transformer, speech and experimental vision models on local hardware. The default paths are pure Go: CPU execution uses checked AVX2, NEON and RVV kernels with scalar fallbacks, while NVIDIA support loads PTX through the driver API without CGo or a CUDA toolkit.

The repository is deliberately broad -- it includes production-shaped LLM and speech paths alongside work-in-progress model families and embedded accelerator experiments -- so the documentation separates runnable features from engineering notes and historical snapshots.

## Try it

Download a small MLX Qwen checkpoint:

```bash
mkdir -p models/qwen3-0.6b
for f in config.json model.safetensors tokenizer.json; do
  curl -L "https://huggingface.co/mlx-community/Qwen3-0.6B-4bit/resolve/main/$f" \
    -o "models/qwen3-0.6b/$f"
done
```

Run it on CPU or NVIDIA:

```bash
# AVX2/NEON with checked scalar fallbacks
go run ./cmd/llm/llmgen \
  -model models/qwen3-0.6b \
  -tokens 50 \
  -prompt "The meaning of life is"

# Runtime-loaded PTX; no CUDA toolkit required
go run ./cmd/llm/llmgen \
  -gpu \
  -model models/qwen3-0.6b \
  -tokens 50 \
  -prompt "The meaning of life is"
```

Interactive chat and the OpenAI-compatible server use the same model loader:

```bash
go run ./cmd/llm/llmchat -model models/qwen3-0.6b -gpu -n 256
go run ./cmd/llm/llmserver -model models/qwen3-0.6b -gpu -listen :8080
```

## Speech

There are two native speech paths:

* `cmd/audio/diarize-vtt` runs Whisper transcription or translation and can produce resumable WebVTT with optional speaker labels. It accepts ordinary media through ffmpeg.
* `cmd/audio/moss-transcribe` runs the pinned MOSS-Transcribe-Diarize graph end to end, including recording-local speaker labels and timestamps. Its verified RTX 3060 path is 2.18x faster than the final forced-CPU path on the JFK fixture.

```bash
# Translate Spanish audio to English WebVTT
go run ./cmd/audio/diarize-vtt \
  -input meeting.m4a \
  -output meeting.vtt \
  -language es

# Native MOSS transcription and diarisation from 16kHz mono PCM WAV
make moss-transcribe
bin/moss-transcribe \
  -model-dir models/MOSS-Transcribe-Diarize \
  -audio meeting.wav \
  -format srt \
  -output meeting.srt
```

See [Whisper and translated VTT](docs/whisper-diarize-vtt.md) and [MOSS transcription and diarisation](docs/moss-transcribe-diarize.md) for model assets, limits and parity gates.

## What runs today

The compact support matrix is in [Supported models](docs/supported-models.md). In practical terms:

* LLaMA, Qwen and Gemma-family decoders cover dense and MoE checkpoints in MLX, GPTQ, BF16/F16/F32 and selected GGUF layouts.
* BERT/GTE encoders, Whisper and MOSS have native inference paths.
* DiffusionGemma and Ideogram 4 have runnable native slices with explicit model-specific limits.
* MiniCPM-V/O, Qwen3-TTS, LFM2, Hunyuan3D, Trellis2 and Z-Image vary from metadata/processor support to partial runtime execution; their support pages state the exact boundary.

Backend selection is automatic where it is safe. `-gpu` selects NVIDIA for the general LLM commands; model-specific commands document their own switches and CPU override. See [Backend selection](docs/backend-selection.md) and [Tuning](docs/tuning.md) before changing cache, placement or worker settings.

## Find the right document

The [documentation index](docs/README.md) is organised by task rather than by implementation history. Useful starting points are:

* [Commands](docs/commands.md) for CLI examples and validation targets.
* [Supported models](docs/supported-models.md) for architecture and format boundaries.
* [Architecture](docs/architecture.md) for the package graph and execution path.
* [Backend stack](docs/backend-stack.md) for CPU, NVIDIA, Vulkan and embedded ownership.
* [Validation gates](docs/validation-gates.md) for the checks required before an optimised path becomes a default.
* [Performance](docs/performance.md) for current benchmarks; the [Gemma4 CPU performance-gap programme](benchmarks/gemma4-gap/README.md) and [CPU SIMD gap](docs/gemma4-cpu-simd-gap.md) freeze the exact E4B CPU oracle; [matmul results](docs/matmul-optimisation-results.md) contains the latest cross-backend optimisation programme.
* [MTP and speculative decoding](docs/mtp-speculative.md) for the current Gemma/Qwen work.

Deep parity investigations, generated snapshots and chronological logs are still available, but they live under the "History and diagnostics" section of the index rather than competing with current guidance.

## Build and test

```bash
go test ./...
go vet ./...
```

Hardware-gated and real-checkpoint tests are opt-in; [Validation gates](docs/validation-gates.md) lists the required environment variables and serialisation rules. The broad suite also contains deliberately documented red or asset-dependent gates, so use the focused target for the backend or model you are changing before treating a failure as a regression.

## License

MIT
