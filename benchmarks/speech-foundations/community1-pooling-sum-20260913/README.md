# Community-1 StatsPool float32 reduction — 13 September 2026

The trained embedding audit exposed a source-contract mismatch in masked support and weighted statistics. Go accumulated resized masks and weighted products serially, while the pinned PyTorch CPU `sum` uses an explicit vector/cascade reduction tree. The difference was three float32 ULP for the 63-frame trained mask (`36.00001144` versus `36.0`) and failed the fixed `1e-6` support gate.

`torchSumF32` now implements the pinned contiguous float32 reduction order in portable Go: eight scalar lanes, four interleaved vector rows, the four-level ATen cascade, scalar remainder first and final lane reduction. It is used for mask sums, squared-weight sums, weighted feature sums and weighted squared-deviation sums. The arithmetic order is explicit and architecture-independent; it does not depend on host SIMD dispatch.

Evidence:

- pinned PyTorch source commit `08187d9e0fba026dc8217405802ab5381dc88d90`;
- `aten/src/ATen/native/cpu/SumKernel.cpp` SHA-256 `9b88aa14e626dd708e50816a5ca89acfc66798226dc82501cf1857caebe96f95`;
- all committed synthetic StatsPool/ResNet oracle tests pass, including bit-exact weight sums;
- all trained masked support values are now bit-exact in both reference-Fbank and Go-Fbank paths; strict failure counts fall from 19/21/23/25 to 17/19/21/23;
- trained endpoint statistics and embeddings remain inside the unchanged `2e-4` gate;
- strict intermediate ResNet traces still contain backend-dependent convolution-reduction differences and are not reclassified by this fix.

No model, tolerance, media/backend selection, service, deployment or GPU path changed.
