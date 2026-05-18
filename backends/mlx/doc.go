// Package mlx contains backend-neutral helpers for MLX-style affine quantized
// weights.
//
// GPU execution for MLX quantized tensors is backend-specific. The NVIDIA
// implementation lives in backends/nvidia because it owns CUDA memory, module
// loading, and kernel dispatch. This package is reserved for MLX format and
// placement abstractions that are independent of a concrete device backend.
package mlx
