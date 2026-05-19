# Quant compatibility import audit

`runtime/quant` is now a compatibility package only. Backend-owned code should import the implementation packages directly.

## Direct backend imports completed

- `model/qwen` now imports `backends/simd/runtime/nvfp4` directly for Qwen3.5 NVFP4 weights and CPU verification/fallback.
- `backends/nvidia/runtime` now imports `backends/simd/runtime/nvfp4` directly for NVFP4 upload validation and CPU dequant fallback.
- CPU hot-path benchmarks, `model/llama.go`, `model/forward_layer.go`, `model/moe.go`, `model/moe_gpu.go`, and `model/gpu_forward.go` now call owning Q4/MLX backend packages directly instead of `runtime/quant` compatibility wrappers.

## Remaining `runtime/quant` imports

These are model-level compatibility call sites and should be retired only after the shared model structs stop exposing compatibility aliases:

- `model/llama_types.go` holds public/shared quantized MLX field aliases.
- `model/gemma4/*` diagnostics follow the same shared model compatibility types.

## Import-boundary check

`runtime/quant/import_boundary_test.go` prevents new backend-owned code from importing `runtime/quant` and keeps the compatibility package limited to its current wrapper files.

## Next cleanup step

Move shared model quantized field types from `runtime/quant` aliases to owning backend packages (`backends/mlx`, `backends/simd/runtime/q4`, `backends/simd/runtime/nvfp4`) in one coordinated model-API change, then leave `runtime/quant` as legacy re-export wrappers only.
