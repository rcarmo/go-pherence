package kernels

import "math"

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
		dst[i] = geluTanh(a[i])
	}
}

func GELUTanhMul(dst, a, b []float32) {
	n := min3(len(dst), len(a), len(b))
	for i := 0; i < n; i++ {
		dst[i] = geluTanh(a[i]) * b[i]
	}
}

func geluTanh(x float32) float32 {
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
