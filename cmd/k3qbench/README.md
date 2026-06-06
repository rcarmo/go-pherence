# k3qbench — quantization-path benchmark

Benchmarks per-quant-type matmul throughput (Q4_K, Q4_0, Q8_0, …) on the K3 to
quantify how each format maps onto the IME2 tile kernels.

## go-pherence packages used
- `backends/k3`, `loader/gguf`, `model`

## Kernels / SIMD to migrate
- None inline; the quant kernels it measures belong in `backends/ggmlquant` and
  `backends/spacemit/ime2` (repack + `vmadotQ4K…`).
