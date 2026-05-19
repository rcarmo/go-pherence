# Refactor plan

This document records the completed backend/model source-tree reorganization and the remaining cleanup direction.

## Completed ownership moves

- Loader boundaries live under `loader/` (`config`, `tokenizer`, `safetensors`, `weights`).
- Backend-neutral placement policy lives under `backends/placement`.
- NVIDIA backend is split into:
  - `backends/nvidia/runtime` for driver/runtime dispatch, buffers, module loading, streams, stats, and kernel wrappers.
  - `backends/nvidia/ptx` for raw PTX strings.
  - `backends/nvidia/ptx/{bf16,q4,mlx,nvfp4}` for quantization-specific PTX.
  - `backends/nvidia/ioctl` for experimental direct NVIDIA ioctl work.
- SIMD backend is split into:
  - `backends/simd/runtime` for public SIMD dispatch wrappers and assembly/scalar fallback selection.
  - `backends/simd/kernels` for reusable CPU kernel bodies split by inference primitive.
  - `backends/simd/runtime/{bf16,q4,nvfp4}` for quantization-specific CPU/SIMD operations.
- MLX format helpers live in `backends/mlx`; NVIDIA execution of MLX weights remains in `backends/nvidia/runtime`.
- Vulkan scaffolding lives under `backends/vulkan`.
- Qwen-specific model code lives under `model/qwen`.
- Gemma4 diagnostic assets live under `model/gemma4`.
- LLaMA-specific primitive helpers that can be separate without owning `LlamaModel` methods live under `model/llama`.
- `runtime/quant` is now a compatibility wrapper layer over backend-owned quantization packages.

See also:

- [backend-layout.md](backend-layout.md)
- [kernel-coverage.md](kernel-coverage.md)
- [architecture.md](architecture.md)

## Current validation gate

Use the workspace temp directory on this container:

```sh
GOTMPDIR=/workspace/tmp/go-pherence-gotmp go test ./...
GOTMPDIR=/workspace/tmp/go-pherence-gotmp go vet ./...
```

Focused package checks for reorg work:

```sh
GOTMPDIR=/workspace/tmp/go-pherence-gotmp go test ./backends/... ./model/... ./runtime/quant ./tensor
```

## Remaining cleanup direction

1. Keep reducing compatibility wrappers once call sites have moved to backend-owned packages.
2. Continue splitting large model files into architecture-owned packages only when Go package boundaries are explicit and tests can move with the code.
3. Add import-boundary checks once the package names stabilize further.
4. Expand backend-specific test coverage when adding AVX2/NEON/NVIDIA kernels.
5. Keep every mechanical move paired with `gofmt` and `go test ./...`.
