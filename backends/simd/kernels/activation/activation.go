package activation

import "math"

const (
	// ActivationReferenceTolerance is the scalar/reference tolerance used for
	// exact Go math paths and golden tests.
	ActivationReferenceTolerance float64 = 1e-6
	// ActivationApproxTolerance is the maximum absolute error budget for future
	// AVX2/NEON polynomial approximations against the scalar reference. Keep this
	// explicit so optimized paths cannot silently loosen correctness expectations.
	ActivationApproxTolerance float64 = 1e-4
)

func SiLU(dst, a []float32) {
	n := len(dst)
	if len(a) < n {
		n = len(a)
	}
	for i := 0; i < n; i++ {
		x := a[i]
		dst[i] = x / (1.0 + float32(math.Exp(float64(-x))))
	}
}

func SiLUMul(dst, a, b []float32) {
	n := min3(len(dst), len(a), len(b))
	for i := 0; i < n; i++ {
		x := a[i]
		s := x / (1.0 + float32(math.Exp(float64(-x))))
		dst[i] = s * b[i]
	}
}

func GELUTanh(dst, a []float32) {
	n := len(dst)
	if len(a) < n {
		n = len(a)
	}
	for i := 0; i < n; i++ {
		dst[i] = GELUTanhScalar(a[i])
	}
}

func GELUTanhMul(dst, a, b []float32) {
	n := min3(len(dst), len(a), len(b))
	for i := 0; i < n; i++ {
		dst[i] = GELUTanhScalar(a[i]) * b[i]
	}
}

func GELUTanhScalar(x float32) float32 {
	x3 := x * x * x
	inner := float32(0.7978845608) * (x + 0.044715*x3)
	return 0.5 * x * (1.0 + float32(math.Tanh(float64(inner))))
}

func min3(a, b, c int) int {
	if b < a {
		a = b
	}
	if c < a {
		a = c
	}
	return a
}
