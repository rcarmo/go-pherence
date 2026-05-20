# Runtime/backend tree move table

Applied by `scripts/mass_move_project_tree.py` batch 4.

| Concern | Target |
|---|---|
| MLX format/types/validation/F16 helpers | `backends/mlx/format` |
| MLX loading | `backends/mlx/loader` |
| MLX dequant/GEMV/GEMM ops | `backends/mlx/ops` |
| MLX runtime capabilities | `backends/mlx/runtime` |
| Quant compatibility Q4/GPTQ wrappers | `runtime/quant/q4` |
| Quant compatibility MLX wrappers | `runtime/quant/mlx` |
| Quant compatibility NVFP4 wrappers | `runtime/quant/nvfp4` |
| Quant compatibility/boundary tests | `runtime/quant/{compat,boundary}` |
| KV cache | `runtime/kv/cache` |
| KV staging | `runtime/kv/staging` |
| KV TurboQuant | `runtime/kv/turboquant` |
| BERT core/workspace/checks | `models/bert/core` |
| BERT fast forward | `models/bert/forward` |
| BERT boundary tests | `models/bert/boundary` |
