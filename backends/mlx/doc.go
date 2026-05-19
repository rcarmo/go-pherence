// Package mlx contains backend-neutral helpers for MLX-style affine quantized
// weights.
//
// MLX affine 4-bit format:
//
//   - weight[outDim, inDim/8] uint32 — 8 packed 4-bit values per uint32,
//     LSB-first
//   - scales[outDim, inDim/group_size] float32/float16/bfloat16 — per-group
//     scale
//   - biases[outDim, inDim/group_size] float32/float16/bfloat16 — per-group
//     bias
//
// Dequantization:
//
//	for each group g in row r:
//	  for each element e in group:
//	    val = (packed >> (e*4)) & 0xF
//	    weight[r][g*group_size + e] = val * scales[r][g] + biases[r][g]
//
// Key differences from GPTQ:
//
//   - Layout is [outDim, inDim/8], not [inDim/8, outDim].
//   - Bias is additive instead of a subtractive zero-point.
//   - No g_idx permutation is used; groups are sequential.
//   - Embeddings may also be quantized.
//
// GPU execution for MLX quantized tensors is backend-specific. The NVIDIA
// implementation lives in backends/nvidia because it owns CUDA memory, module
// loading, and kernel dispatch. This package is reserved for MLX format and
// placement abstractions that are independent of a concrete device backend.
package mlx
