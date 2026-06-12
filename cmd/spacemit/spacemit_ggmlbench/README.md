# spacemit_ggmlbench — GGML compute/quant backend benchmark

Benchmarks the GGML compute + graph + quant paths end-to-end on a GGUF model,
comparing go-pherence's GGML wiring against the vendor baseline.

## go-pherence packages used
- `backends/ggmlcompute`, `backends/ggmlgraph`, `backends/ggmlquant`, `backends/k3`, `loader/gguf`, `model`

## Kernels / SIMD to migrate
- None inline; exercises GGML backend packages. Keep as their benchmark harness.
