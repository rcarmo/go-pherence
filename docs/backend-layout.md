# Backend layout

The repository now uses backend-first ownership with operation/quantization subpackages where Go package boundaries allow it.

## Top-level backend namespaces

| Path | Purpose |
|---|---|
| `backends/nvidia` | NVIDIA backend namespace only. Runtime code is in `runtime`, PTX strings are in `ptx`, direct ioctl experiments are in `ioctl`. |
| `backends/nvidia/runtime` | NVIDIAIDIA driver loading, device buffers, module loading, stream/stats support, and runtime dispatch wrappers for BF16, Q4/GPTQ, MLX, NVFP4, LM-head, RoPE, attention, and SGEMM. Package name remains `nvidia`. |
| `backends/nvidia/ptx` | Raw PTX source constants for non-quantized inference primitives. Quantized PTX is grouped below. |
| `backends/nvidia/ptx/bf16` | BF16 and native-BF16 PTX source constants. |
| `backends/nvidia/ptx/q4` | Q4/GPTQ PTX source constants. |
| `backends/nvidia/ptx/mlx` | MLX affine PTX source constants for NVIDIA execution. |
| `backends/nvidia/ptx/nvfp4` | NVFP4 PTX source constants. |
| `backends/nvidia/ioctl` | Experimental direct NVIDIA ioctl backend. |
| `backends/simd` | SIMD backend namespace only. Runtime code is in `runtime`, shared kernel bodies are in `kernels`. |
| `backends/simd/runtime` | CPU SIMD dispatch facade and assembly/scalar fallback wrappers. Package name remains `simd`. |
| `backends/simd/runtime/bf16` | BF16 CPU/SIMD quantized operations. |
| `backends/simd/runtime/q4` | Q4/GPTQ CPU/SIMD validation (`validate.go`), dequantization (`dequant.go`), GEMV (`gemv.go`), and F16 helpers (`f16.go`). |
| `backends/simd/runtime/nvfp4` | NVFP4 CPU reference types, validation, FP4/F8 decode, dequantization, and GEMV split into focused files. |
| `backends/simd/kernels` | Backend-neutral CPU kernel bodies for attention, RoPE, softmax, layernorm, GELU, and shape helpers. |
| `backends/mlx` | MLX affine quantization format/backend helpers split across types, validation, loading, dequantization, GEMV, and F16 helpers. NVIDIA execution of MLX tensors remains under `backends/nvidia/runtime`. |
| `backends/vulkan` | Vulkan/SPIR-V scaffolding and buffers. |
| `backends/placement` | Backend-neutral memory/placement policy helpers. |

## Compatibility packages

`runtime/quant` is now a compatibility layer for older model call sites. It re-exports or delegates to backend-owned quantization implementations:

- MLX → `backends/mlx`
- Q4/GPTQ → `backends/simd/runtime/q4`
- NVFP4 → `backends/simd/runtime/nvfp4`

New code should prefer backend packages directly unless it deliberately needs the compatibility API. See `docs/quant-import-audit.md` for the current remaining compatibility imports.

## Model layout

| Path | Purpose |
|---|---|
| `model` | Shared LLaMA-family model types, loader, generation, and compatibility wrappers. |
| `model/llama` | LLaMA-specific inference primitives that can be split from the shared model package without owning `LlamaModel` methods. |
| `model/qwen` | Qwen3.5/Qwen3.6 base model, native MTP, safetensors source helpers, NVFP4 GPU cache, and Qwen-specific tests. |
| `model/gemma4` | Gemma4 diagnostic tests and Gemma4-specific investigation assets. |

## Test entrypoints

`make test` tracks the current backend layout and runs loader, model, BERT, NVIDIA, placement, SIMD, Vulkan, runtime, and tensor packages. Use `GOTMPDIR=$PWD/.gotmp go test ./...` for the full repository sweep in constrained `/tmp` environments.

## Naming conventions

- Backend runtime wrappers live in `runtime` subpackages.
- Raw NVIDIA PTX strings live in `ptx` and are grouped by quantization where applicable.
- CPU SIMD reusable kernel bodies live in `backends/simd/kernels` and are split by inference primitive.
- Quantization-specific code should live under the backend that owns execution or format semantics, not in the shared model package.
