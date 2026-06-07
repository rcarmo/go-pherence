# k3ggmlplan — GGML execution plan dump/bench

Builds and inspects the GGML execution plan (`backends/ggmlexec`) for a model to
verify op scheduling and tensor placement before running compute.

## go-pherence packages used
- `backends/ggmlexec`, `backends/k3`, `model`

## Kernels / SIMD to migrate
- None inline; planning/inspection only.
