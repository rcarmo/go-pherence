# k3ortlayerbench — per-layer ORT vs native comparison

Runs individual transformer layers through both the SpaceMIT ORT EP and the k3
native path to compare per-op throughput and pinpoint where ORT wins/loses.

## go-pherence packages used
- `backends/k3`, `backends/spacemit/ort`, `loader/gguf`

## Kernels / SIMD to migrate
- None inline; comparison harness over both backends.
