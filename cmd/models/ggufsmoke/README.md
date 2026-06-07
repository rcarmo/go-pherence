# ggufsmoke — GGUF load + single-forward smoke test

Loads a GGUF through `loader/gguf` into a `model`, runs one forward pass on the
k3 backend, and checks logits are finite — fast "does this model load+run" gate.

## go-pherence packages used
- `backends/k3`, `loader/gguf`, `model`, `runtime/kv`

## Kernels / SIMD to migrate
- None inline; smoke harness over backend + model.
