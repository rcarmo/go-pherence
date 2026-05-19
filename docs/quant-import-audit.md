# Quant compatibility import audit

`runtime/quant` is now a compatibility package only. Backend-owned code imports implementation packages directly.

## Direct backend imports completed

- `model/qwen` imports `backends/simd/runtime/nvfp4` directly for Qwen3.5 NVFP4 weights and CPU verification/fallback.
- `backends/nvidia/runtime` imports `backends/simd/runtime/nvfp4` directly for NVFP4 upload validation and CPU dequant fallback.
- CPU hot-path benchmarks and model execution/loader paths now call owning Q4/MLX backend packages directly:
  - `backends/mlx`
  - `backends/simd/runtime/q4`
  - `backends/simd/runtime/nvfp4`
- Shared model MLX field types now use `backends/mlx.QuantWeight` directly.
- Gemma4 diagnostics now use `backends/mlx.Gemv` directly.

## Remaining `runtime/quant` imports

No production/model/backend Go package imports `runtime/quant` directly. The only textual references are inside `runtime/quant` itself, including its import-boundary tests.

## Import-boundary check

`runtime/quant/import_boundary_test.go` prevents new backend-owned code from importing `runtime/quant` and keeps the compatibility package limited to its current wrapper files.

## Next cleanup step

Keep `runtime/quant` as legacy re-export wrappers for external callers. Remove wrapper files only in a deliberate public API cleanup.
