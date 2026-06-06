# qwenmtpsmoke — Qwen MTP smoke test

Loads the Qwen MTP head and runs a minimal forward pass to confirm wiring.
MTP was found non-viable on the K3 CPU path (see review); kept as evidence.

## go-pherence packages used
- `loader/config`, `model/qwen`

## Kernels / SIMD to migrate
- None inline.
