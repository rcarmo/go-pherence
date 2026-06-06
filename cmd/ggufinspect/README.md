# ggufinspect — GGUF metadata/tensor inspector

Dumps GGUF header, KV metadata, tensor list (names, shapes, dtypes) and KV-cache
sizing for a model. First stop when onboarding a new GGUF.

## go-pherence packages used
- `loader/gguf`, `runtime/kv`

## Kernels / SIMD to migrate
- None inline; inspection only.
