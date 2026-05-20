package simd

import "github.com/rcarmo/go-pherence/backends/simd/kernels"

// SiLU computes dst[i] = a[i] / (1 + exp(-a[i])).
func SiLU(dst, a []float32) { kernels.SiLU(dst, a) }

// SiLUTo computes SiLU into caller-owned output and reports malformed inputs.
func SiLUTo(dst, a []float32) bool {
	if len(dst) == 0 || len(a) < len(dst) {
		return false
	}
	kernels.SiLU(dst, a)
	return true
}

// SiLUMul computes dst[i] = silu(a[i]) * b[i].
// It is the activation-owned alias for the legacy VecSiLUMul entrypoint.
func SiLUMul(dst, a, b []float32) { VecSiLUMul(dst, a, b) }

// SiLUMulTo computes SiLU×Mul into caller-owned output and reports malformed inputs.
func SiLUMulTo(dst, a, b []float32) bool {
	if len(dst) == 0 || len(a) < len(dst) || len(b) < len(dst) {
		return false
	}
	VecSiLUMul(dst, a, b)
	return true
}

// GELUTanh computes dst[i] = gelu_tanh(a[i]).
func GELUTanh(dst, a []float32) { kernels.GELUTanh(dst, a) }

// GELUTanhChecked allocates output for GELU(tanh) and reports malformed inputs.
func GELUTanhChecked(a []float32) ([]float32, bool) {
	if len(a) == 0 {
		return nil, false
	}
	out := make([]float32, len(a))
	kernels.GELUTanh(out, a)
	return out, true
}

// GELUTanhTo computes GELU(tanh) into caller-owned output and reports malformed inputs.
func GELUTanhTo(dst, a []float32) bool {
	if len(dst) == 0 || len(a) < len(dst) {
		return false
	}
	kernels.GELUTanh(dst, a)
	return true
}

// GELUTanhMulTo computes GELU(tanh)×Mul into caller-owned output and reports malformed inputs.
func GELUTanhMulTo(dst, a, b []float32) bool {
	if len(dst) == 0 || len(a) < len(dst) || len(b) < len(dst) {
		return false
	}
	GELUTanhMul(dst, a, b)
	return true
}

// GELUTanhScalar computes the tanh-approximation GELU for one value.
func GELUTanhScalar(x float32) float32 { return kernels.GELUTanhScalar(x) }
