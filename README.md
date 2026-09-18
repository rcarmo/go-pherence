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

For library use, `github.com/rcarmo/go-pherence/loader/audio/media` exports `media.NewGo264(media.Go264Config{})`: a pure-Go `media.Adapter` for PCM WAV and the documented progressive AAC-LC MP4/M4A subset. It emits canonical 16 kHz mono S16 WAV and works with `media.OpenCanonicalPCM`, without FFmpeg, model weights, cgo or special build tags. See [import example and format limits](docs/speech/go264-media.md). The MIT provider `github.com/rcarmo/go-264/audio` can also be imported directly. Existing CLI and speech-job defaults still use FFmpeg; the pure-Go adapter is selected explicitly.

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

See [Whisper and translated VTT](docs/speech/whisper-diarize-vtt.md) and [MOSS transcription and diarisation](docs/speech/moss-transcribe-diarize.md) for model assets, limits and parity gates.

## What runs today

The compact support matrix is in [Supported models](docs/models/supported-models.md). In practical terms:

* LLaMA, Qwen and Gemma-family decoders cover dense and MoE checkpoints in MLX, GPTQ, BF16/F16/F32 and selected GGUF layouts.
* BERT/GTE encoders, Whisper and MOSS have native inference paths.
* [Jevlike](model/jevlike/README.md) provides tiny/frozen choice scoring, training, evaluation and reusable visual/action adapters. Game environments, recordings and demo orchestration are outside the port.
* [GLiNER 2.5](model/gliner2/README.md) runs native entity, classification, relation, record and mixed-schema inference against the published checkpoint. Its guide distinguishes the supported API from broader upstream compatibility.
* DiffusionGemma and Ideogram 4 have runnable native slices with explicit model-specific limits.
* MiniCPM-V/O, Qwen3-TTS, LFM2, Hunyuan3D, Trellis2 and Z-Image vary from metadata/processor support to partial runtime execution; their support pages state the exact boundary.

Backend selection is automatic where it is safe. `-gpu` selects NVIDIA for the general LLM commands; model-specific commands document their own switches and CPU override. See [Backend selection](docs/backends/backend-selection.md) and [Tuning](docs/guides/tuning.md) before changing cache, placement or worker settings.

## Find the right document

The [documentation index](docs/README.md) is organised by task rather than by implementation history. Useful starting points are:

* [Commands](docs/guides/commands.md) for CLI examples and validation targets.
* [Supported models](docs/models/supported-models.md) for architecture and format boundaries.
* [Architecture](docs/architecture/architecture.md) for the package graph and execution path.
* [Backend stack](docs/architecture/backend-stack.md) for CPU, NVIDIA, Vulkan and embedded ownership.
* [Validation gates](docs/validation/validation-gates.md) for the checks required before an optimised path becomes a default.
* [Performance](docs/performance/performance.md) for current benchmarks; the [Gemma4 CPU performance-gap programme](benchmarks/gemma4-gap/README.md) and [CPU SIMD gap](docs/performance/gemma4-cpu-simd-gap.md) freeze the exact E4B CPU oracle; [matmul results](docs/performance/matmul-optimisation-results.md) contains the latest cross-backend optimisation programme.
* [MTP and speculative decoding](docs/models/mtp-speculative.md) for the current Gemma/Qwen work.

Current guidance lives in topic folders under `docs/`; dated investigations and chronological logs are indexed in [history](docs/history/README.md). Provenance and package-specific validation remain beside their implementations.

## Build and test

```bash
make host-build
make host-vet
make host-test

# Compile Linux/RISC-V K3 code and test binaries without executing them
make spacemit-cross-compile
```

Host checks respect Go build constraints: Linux/RISC-V AICPU and TCM execution is not forced into amd64 or ARM64 tests, while portable packing and scalar fallbacks remain checked. Cross-build success is compilation evidence only. [Validation gates](docs/validation/validation-gates.md) lists hardware and asset requirements and the remaining whole-tree failures; the host targets do not hide unrelated errors.

## License

MIT
