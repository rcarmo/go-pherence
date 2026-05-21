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

# Diagnostic package import/build boundary, when local diagnostic assets permit it
GOTMPDIR=$PWD/.gotmp go test -tags diagnostic ./model/gemma4

# Gemma4 31B packed MTP smoke, when local ignored model assets are present
GOTMPDIR=$PWD/.gotmp go run ./cmd/llmgen \
  -model models/gemma4-31b-it-4bit \
  -mtp-drafter models/gemma4-31b-it-mtp-assistant-4bit \
  -mtp-smoke \
  -prompt "Hello"
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
