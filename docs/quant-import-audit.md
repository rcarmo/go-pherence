# Quant compatibility import audit

`runtime/quant` is now a compatibility package only. Backend-owned code should import the implementation packages directly.

## Direct backend imports completed

- `model/qwen` now imports `backends/simd/runtime/nvfp4` directly for Qwen3.5 NVFP4 weights and CPU verification/fallback.
- `backends/nvidia/runtime` now imports `backends/simd/runtime/nvfp4` directly for NVFP4 upload validation and CPU dequant fallback.

## Remaining `runtime/quant` imports

These are model-level compatibility call sites and should be retired only after the shared model structs stop exposing compatibility aliases:

- `model/llama.go` and `model/llama_types.go` hold public/shared quantized weight fields.
- `model/forward_layer.go`, `model/gpu_forward.go`, `model/moe.go`, and `model/moe_gpu.go` operate on those shared fields.
- `model/cpu_hotpath_bench_test.go`, `model/moe_gpu_test.go`, and `model/gemma4/*` diagnostics follow the same shared model compatibility types.

## Next cleanup step

Move shared model quantized field types from `runtime/quant` aliases to owning backend packages (`backends/mlx`, `backends/simd/runtime/q4`, `backends/simd/runtime/nvfp4`) in one coordinated model-API change, then leave `runtime/quant` as legacy re-export wrappers only.
