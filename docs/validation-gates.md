# Validation gates

This repository uses phase-level validation for backend coverage work. Do not run the full test/vet matrix after every small mechanical change; run it when a complete plan phase is ready to validate.

## Standard phase gate

Use the workspace-local Go temp directory in constrained containers:

```bash
GOTMPDIR=$PWD/.gotmp go test ./...
GOTMPDIR=$PWD/.gotmp go vet ./...
make test-cpu
```

`make test-cpu` keeps NVIDIA disabled and Vulkan software devices opt-in only, so CPU/scalar fallback behavior is checked without requiring GPU hardware.

## Optional phase-specific gates

Run these only when the corresponding phase is being accepted:

```bash
# SIMD/CPU backend phases
GOTMPDIR=$PWD/.gotmp go test ./backends/simd/... ./backends/mlx ./model

# NVIDIA runtime phase on CPU-only hosts
GOTMPDIR=$PWD/.gotmp go test ./backends/nvidia/runtime ./backends/nvidia/ioctl

# Vulkan wrapper phase without a Vulkan device
GOTMPDIR=$PWD/.gotmp go test ./backends/vulkan

# Whisper large-v3-turbo execution graph (audio commands, mel, SIMD, prompt/timestamp/speculative scaffolds).
# Treat the native CPU/SIMD path as the oracle for NEON, RISC-V/IME, A100,
# NVIDIA, and CUDA/PTX refinements; hardware paths must preserve these results.
make whisper-turbo-check

# Equivalent explicit form:
GOTMPDIR=$PWD/.gotmp go test \
  ./models/whisper \
  ./cmd/audio/... \
  ./loader/audio \
  ./backends/simd/fft \
  ./backends/simd/runtime \
  ./backends/cuda/ptx
python3 scripts/whisper_turbo_smoke.py --audio testdata/jfk.wav
# whisper_turbo_smoke covers standalone translate/transcribe, standalone chunked
# no-timestamp, standalone timestamp VTT for translate/transcribe, standalone timestamp+diarize VTT,
# diarize-vtt translate/transcribe, and diarize-vtt+speaker for both translate
# and transcribe.
python3 scripts/speakercheck_suite.py testdata/speakercheck_suite.json

# Optional A100 row-scale FFN default-candidate parity check; useful on K3/A100 hosts,
# harmless on non-riscv hosts where A100 stubs keep the path disabled. Compares
# plain stdout decode for translate/transcribe, standalone timestamp/VTT output for translate/transcribe, diarize-vtt output, and diarize-vtt with speaker labels for translate/transcribe.
# Pass repeated --audio flags and optional --start/--duration to
# scripts/whisper_a100_compare.py for broader/long-form clip sets.
make whisper-a100-compare
make whisper-a100-podcast-compare
# podcast target compares both stdout and timestamp/VTT on a 12s long-form window.

# Optional K3/RISC-V int8/IME parity check against the native SIMD oracle.
# On non-riscv hosts, int8 stubs keep this a command/prompt smoke.
make whisper-int8-compare

# Diagnostic package import/build boundary, when local diagnostic assets permit it
GOTMPDIR=$PWD/.gotmp go test -tags diagnostic ./model/gemma4

# Gemma4 31B packed MTP smoke, when local ignored model assets are present
GOTMPDIR=$PWD/.gotmp go run ./cmd/llm/llmgen \
  -model models/gemma4-31b-it-4bit \
  -mtp-drafter models/gemma4-31b-it-mtp-assistant-4bit \
  -mtp-smoke \
  -prompt "Hello"

# Recommended real-prompt Gemma4 E4B smoke on the local RTX 3060 profile
GOTMPDIR=$PWD/.gotmp go run ./cmd/llm/llmgen \
  -gpu -gpu-layers 0 \
  -model models/gemma4-e4b-it-4bit \
  -mtp-drafter models/gemma4-e4b-mtp-drafter \
  -mtp-smoke -mtp-real-prompt \
  -prompt "Hi"

# 31B stress smoke, when VRAM headroom permits
GOTMPDIR=$PWD/.gotmp go run ./cmd/llm/llmgen \
  -gpu -gpu-layers 17 -gpu-kv-max-seq 256 \
  -model models/gemma4-31b-it-4bit \
  -mtp-drafter models/gemma4-31b-it-mtp-assistant-4bit \
  -mtp-smoke -mtp-real-prompt \
  -prompt "Hi"
```

## Benchmark snapshots

Benchmark snapshots are documentation data, not a per-change requirement. Refresh them at phase validation time with:

```bash
GOTMPDIR=$PWD/.gotmp go test ./model -run '^$' -bench 'BenchmarkCPUHot' -benchmem
```

Update `docs/performance.md` and `docs/cpu-simd-coverage.md` after capturing new snapshots.

## Hardware smoke tests

NVIDIA and Vulkan parity tests should be availability-gated. They should skip cleanly when required hardware, drivers, or opt-in environment variables are absent.

- NVIDIA smoke: validate on a CUDA-capable host with the normal backend enabled.
- Vulkan smoke: validate only after SPIR-V pipeline cache wiring lands; CPU/software Vulkan remains opt-in via `GO_PHERENCE_VULKAN_ALLOW_CPU=1`.
