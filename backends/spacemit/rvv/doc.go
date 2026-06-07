// Package rvv implements low-level RISC-V Vector (RVV 1.0) SIMD kernels for the
// SpaceMIT K3 X60 cores: int8 GEMM (dot and register-blocked outer-product),
// W4A8, byte-copy, and q8 block quantization.
//
// Kernels use WORD-encoded assembly because the Go 1.24 toolchain has no RVV
// assembler. This is a leaf kernel package with no inference-engine
// dependencies.
package rvv
