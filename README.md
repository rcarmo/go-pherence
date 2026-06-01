# go-pherence

![go-pherence](docs/icon-256.png)

**A portable Go inference toolkit for local and embedded AI.** go-pherence runs transformer decoders, encoders, speech models, and experimental accelerator paths from a single Go codebase, with NVIDIA GPU execution, SIMD CPU kernels, MLX/GPTQ/BF16/F16/F32 weight support, Vulkan scaffolding, and embedded SoC backends. It builds as a single static Go binary whenever possible: no Python runtime, no mandatory native SDK, and no CGo dependency in the default hot path.

> Current development focus: practical local inference across NVIDIA, CPU/SIMD, and embedded targets; native MTP/speculative decoding for Gemma4 and Qwen3.6; Whisper decoder acceleration; and SpacemiT K3/RISC-V IME2 experiments. See [docs/mtp-speculative.md](docs/mtp-speculative.md), [docs/qwen36-mtp.md](docs/qwen36-mtp.md), [docs/spacemit-ime2.md](docs/spacemit-ime2.md), and [docs/backend-stack.md](docs/backend-stack.md).

## Why

go-pherence is intended to be a small, inspectable inference stack for running real models on the hardware people actually have: NVIDIA cards with tight VRAM budgets, Intel/AMD laptops, ARM SBCs, and emerging embedded accelerators. It treats model formats such as MLX affine quantization as inputs rather than as the product identity, and keeps backend ownership explicit so optimized paths can coexist with checked scalar fallbacks.

## Quick start

```bash
# Download a small quantized Qwen model
mkdir -p models/qwen3-0.6b
for f in config.json model.safetensors tokenizer.json; do
  curl -L "https://huggingface.co/mlx-community/Qwen3-0.6B-4bit/resolve/main/$f" \
    -o "models/qwen3-0.6b/$f"
done

# CPU path: AVX2/NEON SIMD with scalar fallbacks
go run ./cmd/llmgen -model models/qwen3-0.6b -tokens 50 -prompt "The meaning of life is"

# NVIDIA path: runtime-loaded PTX, zero CGo
go run ./cmd/llmgen -gpu -model models/qwen3-0.6b -tokens 50 -prompt "The meaning of life is"
```

## Current highlights

- **Backends:** NVIDIA PTX runtime, AVX2/NEON SIMD CPU runtime, Vulkan scaffolding, and embedded accelerator experiments such as SpacemiT K3 IME2.
- **Weight formats:** MLX affine 4-bit, GPTQ/Q4, BF16, F16, F32, and model-specific packed paths where useful.
- **Architectures:** LLaMA-family, Qwen2/3/Qwen3Next, Qwen3 MoE, Gemma3/Gemma4, BERT/GTE encoders, Whisper large-v3 translated VTT pipeline, and experimental 3D/vision metadata tooling.
- **Hybrid placement:** `--gpu-layers N`, compact LM-head placement, reusable GPU caches, and planner-driven windowing for models larger than available VRAM.
- **Embedded scenarios:** low-allocation CPU execution, quantized kernels, RISC-V IME2 INT8 matmul work, and static-binary deployment goals.
- **Speculative/MTP work:** Gemma4 assistant loader, packed 4-bit assistant execution, real prompt activation/KV smoke, and Qwen3.6 native-MTP diagnostics.
- **Validation:** checked runtime APIs, import-boundary tests, malformed-input guards, and hardware-gated smoke tests.

## Recommended MTP development target

The fastest local MTP target is the Gemma4 E4B pair:

```text
models/gemma4-e4b-it-4bit
models/gemma4-e4b-mtp-drafter
```

It fits fully on the local RTX 3060 (`42/42` layers resident, compact quantized LM head resident, ~5GiB VRAM free). A 207-token real-prompt MTP smoke prefills in about `9.25s`, and the assistant drafter step is about `0.10s`.

Example:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/llmgen \
  -gpu -gpu-layers 0 \
  -model models/gemma4-e4b-it-4bit \
  -mtp-drafter models/gemma4-e4b-mtp-drafter \
  -mtp-smoke -mtp-real-prompt \
  -prompt "Hi"
```

`-mtp-smoke` is a runtime smoke path, not full speculative generation yet.

## Common commands

```bash
# One-shot generation
go run ./cmd/llmgen -model models/qwen3-0.6b -gpu -tokens 50 -prompt "Hello"

# Interactive chat
go run ./cmd/llmchat -model models/gemma4-e2b-it-4bit -gpu -n 256

# OpenAI-compatible server
go run ./cmd/llmserver -model models/gemma4-e2b-it-4bit -gpu -listen :8080

# Stock-weight speculative benchmark scaffold
go run ./cmd/specbench -model models/smollm2-135m -prompt-file prompts.txt -tokens 16 -repeat 3

# Large-v3 translated WebVTT from audio (GPU-assisted, resumable)
go run ./cmd/diarize-vtt -input meeting.m4a -output meeting.vtt -language es

# Native pure Go/SIMD GGUF REAP/TurboQuant validation and benchmark
make gguf-inspect-qwen36-reap
make gguf-smoke-qwen36-reap
make gguf-validate-qwen36-reap
make gguf-bench-qwen36-reap
make gguf-check-qwen36-reap
make gguf-ci-qwen36-reap
```

See [docs/commands.md](docs/commands.md) for detailed command usage, GGUF REAP/TurboQuant validation, MTP smoke commands, Qwen3.6 native-MTP triage commands, Whisper VTT usage, and benchmark harnesses. See [docs/whisper-diarize-vtt.md](docs/whisper-diarize-vtt.md) for the current Whisper implementation status and limitations.

## Documentation map

Start here:

- [docs/README.md](docs/README.md) — full documentation index.
- [docs/supported-models.md](docs/supported-models.md) — supported architectures, formats, and performance snapshot.
- [docs/commands.md](docs/commands.md) — CLI usage and smoke/benchmark commands.
- [docs/backend-stack.md](docs/backend-stack.md) — NVIDIA, Vulkan, SIMD, BF16, and package ownership summary.
- [docs/mtp-speculative.md](docs/mtp-speculative.md) — Gemma4/Qwen3.6 MTP implementation notes.
- [docs/whisper-diarize-vtt.md](docs/whisper-diarize-vtt.md) — current Whisper translated VTT pipeline and performance notes.
- [docs/gemma4-31b-runbook.md](docs/gemma4-31b-runbook.md) — Gemma4 E4B/31B local run strategy and smoke results.
- [docs/qwen36-mtp.md](docs/qwen36-mtp.md) — Qwen3.6 native-MTP checkpoint findings.
- [docs/validation-gates.md](docs/validation-gates.md) — standard validation gates.
- [docs/validation-hardening.md](docs/validation-hardening.md) — malformed-input and boundary-hardening summary.

## SpacemiT K3 / RISC-V IME2 acceleration

go-pherence includes a pure Go backend for SpacemiT K3 SoC (MilkV Jupiter 2), using IME2 hardware for accelerated INT8 matrix multiply.

### Performance

| Benchmark | Result |
|-----------|--------|
| Raw vmadot (4x8x4 MAC) | 9.3ns / 128M ops/sec |
| Pre-packed GEMM (1024 cubed, 8 threads) | 142 GOPS |
| Single matmul (2048x1024) | 326us |
| End-to-end decode (Qwen3-0.6B) | 14 tok/s (INT8 LM head) |

### Key findings

- IME2 spec: `spacemit-com/riscv-ime-extension-spec`
- GCC flag: `-march=rv64gcv_xsmtvdotii`
- X100 cores are CPU 0-7; A100 efficiency cores are 8-15
- TCM: mmap `/dev/tcm` gives 3MB SRAM
- Q4_K repack is identity (no tile format change)
- Full details: `docs/spacemit-ime2.md`

## License

MIT
