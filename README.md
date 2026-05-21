# go-pherence

![go-pherence](docs/icon-256.png)

**Run MLX models on any hardware.** go-pherence is a pure Go inference engine for Apple MLX, GPTQ, BF16, F16, and F32 model weights, with NVIDIA GPU execution, Vulkan scaffolding, and AVX2/NEON CPU paths. It builds as a single static Go binary: no Python, no CGo, no mandatory native SDK.

> Current development focus: native MTP/speculative decoding for Gemma4 and Qwen3.6, with Gemma4 E4B as the fast local target and 31B/Qwen3.6 as stress paths. See [docs/mtp-speculative.md](docs/mtp-speculative.md), [docs/gemma4-31b-runbook.md](docs/gemma4-31b-runbook.md), and [docs/qwen36-mtp.md](docs/qwen36-mtp.md).

## Why

Apple's [MLX](https://github.com/ml-explore/mlx) ecosystem has an excellent quantized model collection on Hugging Face, but MLX itself targets Apple Silicon. go-pherence runs compatible MLX weights on NVIDIA GPUs, Intel/AMD CPUs, ARM SBCs, and eventually broader Vulkan devices.

## Quick start

```bash
# Download a small MLX model
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

- **Primary format:** MLX affine 4-bit safetensors.
- **Backends:** NVIDIA PTX runtime, SIMD CPU runtime, Vulkan scaffolding.
- **Architectures:** LLaMA-family, Qwen2/3, Qwen3 MoE, Gemma3, Gemma4, and BERT/GTE encoder path.
- **Hybrid placement:** `--gpu-layers N` for partial GPU residency.
- **Large-vocab handling:** compact MLX LM-head placement under tight VRAM.
- **MTP work:** Gemma4 assistant loader, packed 4-bit assistant execution, real prompt activation/KV smoke, and Qwen3.6 native-MTP diagnostics.
- **Validation:** checked runtime APIs, import-boundary tests, malformed-input guards, and hardware-gated smoke tests.

## Recommended MTP development target

The fastest local MTP target is the Gemma4 E4B pair:

```text
models/gemma4-e4b-it-4bit
models/gemma4-e4b-mtp-drafter
```

It fits fully on the local RTX 3060 (`42/42` layers resident, compact MLX LM head resident, ~5GiB VRAM free). A 207-token real-prompt MTP smoke prefills in about `9.25s`, and the assistant drafter step is about `0.10s`.

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
go run ./cmd/llmchat -model models/gemma4-e2b-mlx4 -gpu -n 256

# OpenAI-compatible server
go run ./cmd/llmserver -model models/gemma4-e2b-mlx4 -gpu -listen :8080

# Stock-weight speculative benchmark scaffold
go run ./cmd/specbench -model models/smollm2-135m -prompt-file prompts.txt -tokens 16 -repeat 3
```

See [docs/commands.md](docs/commands.md) for detailed command usage, MTP smoke commands, Qwen3.6 native-MTP triage commands, and benchmark harnesses.

## Documentation map

Start here:

- [docs/README.md](docs/README.md) — full documentation index.
- [docs/supported-models.md](docs/supported-models.md) — supported architectures, formats, and performance snapshot.
- [docs/commands.md](docs/commands.md) — CLI usage and smoke/benchmark commands.
- [docs/backend-stack.md](docs/backend-stack.md) — NVIDIA, Vulkan, SIMD, BF16, and package ownership summary.
- [docs/mtp-speculative.md](docs/mtp-speculative.md) — Gemma4/Qwen3.6 MTP implementation notes.
- [docs/gemma4-31b-runbook.md](docs/gemma4-31b-runbook.md) — Gemma4 E4B/31B local run strategy and smoke results.
- [docs/qwen36-mtp.md](docs/qwen36-mtp.md) — Qwen3.6 native-MTP checkpoint findings.
- [docs/validation-gates.md](docs/validation-gates.md) — standard validation gates.
- [docs/validation-hardening.md](docs/validation-hardening.md) — malformed-input and boundary-hardening summary.

## License

MIT
