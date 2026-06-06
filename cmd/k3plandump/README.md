# k3plandump — model plan / tensor-placement dump

Dumps the resolved k3 execution plan for a model: per-layer op list, tensor
shapes, dtypes, and backend/core placement decisions. Diagnostic for the
placement engine (`backends/placement`).

## go-pherence packages used
- `backends/k3`, `model`

## Kernels / SIMD to migrate
- None inline; inspection only.
