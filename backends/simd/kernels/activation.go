package kernels

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

func GELUExact(dst, a []float32) {
	n := len(dst)
	if len(a) < n {
		n = len(a)
	}
	for i := 0; i < n; i++ {
		dst[i] = GELUExactScalar(a[i])
	}
}

func GELUExactMul(dst, a, b []float32) {
	n := min3(len(dst), len(a), len(b))
	for i := 0; i < n; i++ {
		dst[i] = GELUExactScalar(a[i]) * b[i]
	}
}

func GELUExactScalar(x float32) float32 {
	const invSqrt2 = 0.70710678118654752440
	return 0.5 * x * (1.0 + float32(math.Erf(float64(x)*invSqrt2)))
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
	return 0.5 * x * (1.0 + fastTanhF32(inner))
}

// fastTanhF32 computes tanh(x) in float32 using bit-manipulation exp2.
// Max error < 1e-6 for |x| < 9.
func fastTanhF32(x float32) float32 {
	if x > 9 {
		return 1
	}
	if x < -9 {
		return -1
	}
	// tanh(x) = (exp(2x)-1)/(exp(2x)+1)
	e := fastExpF32(2 * x)
	return (e - 1) / (e + 1)
}

// fastExpF32 computes exp(x) in float32 using the range-reduction identity
// exp(x) = 2^(x/ln2) and a minimax polynomial for 2^frac.
func fastExpF32(x float32) float32 {
	const (
		ln2inv = 1.4426950408889634 // 1/ln(2)
		ln2hi  = 0.693145751953125
		ln2lo  = 1.428606765330187e-06
	)
	if x > 88 {
		return float32(math.Inf(1))
	}
	if x < -88 {
		return 0
	}
	// Range reduction: x = n*ln2 + r, |r| <= ln2/2
	n := int32(x*ln2inv + 0.5)
	if x < 0 {
		n = int32(x*ln2inv - 0.5)
	}
	r := x - float32(n)*ln2hi - float32(n)*ln2lo
	// Minimax polynomial for exp(r)-1 on [-ln2/2, ln2/2]
	r2 := r * r
	p := r + r2*(0.5+r*(0.16666667+r*(0.041666668+r*(0.008333334+r*0.001388889))))
	// exp(x) = 2^n * (1+p)
	result := 1.0 + p
	// Multiply by 2^n via bit manipulation
	bits := math.Float32bits(result)
	bits += uint32(n) << 23
	return math.Float32frombits(bits)
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
