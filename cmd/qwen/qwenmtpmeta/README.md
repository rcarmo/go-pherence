# qwenmtpmeta — Qwen MTP metadata probe

Inspects the multi-token-prediction (MTP) head metadata in Qwen safetensors /
config (extra head tensors, layout) to assess MTP viability. Part of the MTP
investigation that was ultimately rejected on this CPU backend.

## go-pherence packages used
- `loader/config`, `loader/safetensors`

## Kernels / SIMD to migrate
- None inline; metadata inspection.
