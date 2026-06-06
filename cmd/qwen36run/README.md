# qwen36run — Qwen3.6 reference runner

Runs the Qwen3.6 (MoE A3B) model through the go-pherence `model/qwen` path with
pluggable backends (MLX / NVIDIA runtime / CPU), used to validate model
correctness independent of the K3 kernels.

## go-pherence packages used
- `model/qwen`, `backends/mlx`, `backends/nvidia/runtime`, `loader/config`, `loader/tokenizer`, `runtime/kv`, `tensor`

## Kernels / SIMD to migrate
- None inline; model-definition driver. Keep as the Qwen3.6 conformance runner.
