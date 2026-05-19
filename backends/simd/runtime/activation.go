package simd

import "github.com/rcarmo/go-pherence/backends/simd/kernels"

// SiLU computes dst[i] = a[i] / (1 + exp(-a[i])).
func SiLU(dst, a []float32) { kernels.SiLU(dst, a) }

// SiLUMul computes dst[i] = silu(a[i]) * b[i].
// It is the activation-owned alias for the legacy VecSiLUMul entrypoint.
func SiLUMul(dst, a, b []float32) { VecSiLUMul(dst, a, b) }

// GELUTanh computes dst[i] = gelu_tanh(a[i]).
func GELUTanh(dst, a []float32) { kernels.GELUTanh(dst, a) }
