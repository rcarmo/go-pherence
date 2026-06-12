# spacemit_ffnblockbench — FFN block micro-benchmark

Isolates a single transformer FFN block (gate/up/down + activation) and
benchmarks it through the GGML graph backend — the hot path for MoE expert
matmuls on the K3.

## go-pherence packages used
- `backends/ggmlgraph`, `backends/k3`, `loader/gguf`

## Kernels / SIMD to migrate
- None inline; the FFN/gate-fuse kernels it measures live (or should live) in
  `backends/spacemit/ime2` (`q4k_ffn_fuse`, `q4k_gate_fuse` from `ime2run`).
