# spacemit_graphfusebench — fused-graph benchmark

Measures the effect of graph-level operator fusion (e.g. matmul+activation,
gate+up fusion) in the GGML graph backend versus the unfused path.

## go-pherence packages used
- `backends/ggmlgraph`, `backends/k3`, `loader/gguf`

## Kernels / SIMD to migrate
- None inline; fusion logic belongs in `backends/ggmlgraph` / `backends/spacemit/ime2`.
