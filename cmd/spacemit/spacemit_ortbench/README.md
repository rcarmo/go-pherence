# spacemit_ortbench — SpaceMIT ONNX Runtime EP benchmark

Benchmarks the SpaceMIT ONNX Runtime execution provider
(`SpaceMITExecutionProvider`) as an alternative AI-core path to the in-house
IME2 kernels.

## go-pherence packages used
- `backends/spacemit/ort`

## Kernels / SIMD to migrate
- None of ours; compute is inside the closed `libspacemit_ep.so`. Comparison
  baseline for the native IME2 path.
