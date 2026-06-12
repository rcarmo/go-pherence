package rvv

import (
	"math"
	"testing"
)

func exactSiLU(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}

func TestSiLUMulRVVAccuracy(t *testing.T) {
	n := 4096
	a := make([]float32, n)
	b := make([]float32, n)
	dst := make([]float32, n)
	want := make([]float32, n)
	for i := range a {
		a[i] = float32(i-n/2) * 0.01
		b[i] = float32((i*7)%n-n/2) * 0.005
	}
	SiLUMulRVV(dst, a, b)
	for i := range want {
		want[i] = exactSiLU(a[i]) * b[i]
	}
	var maxErr float64
	for i := range dst {
		d := math.Abs(float64(dst[i] - want[i]))
		if d > maxErr {
			maxErr = d
		}
	}
	if maxErr > 0.05 {
		t.Fatalf("maxErr=%g", maxErr)
	}
	t.Logf("maxErr=%g", maxErr)
}

func BenchmarkSiLUMulRVV(b *testing.B) {
	n := 12288 * 16 // DiT intermediate * batch
	a := make([]float32, n)
	bb := make([]float32, n)
	dst := make([]float32, n)
	for i := range a {
		a[i] = float32(i%1000-500) * 0.01
		bb[i] = float32(i%997-498) * 0.01
	}
	b.SetBytes(int64(n * 4 * 3))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SiLUMulRVV(dst, a, bb)
	}
}

func BenchmarkSiLUMulScalar(b *testing.B) {
	n := 12288 * 16
	a := make([]float32, n)
	bb := make([]float32, n)
	dst := make([]float32, n)
	for i := range a {
		a[i] = float32(i%1000-500) * 0.01
		bb[i] = float32(i%997-498) * 0.01
	}
	b.SetBytes(int64(n * 4 * 3))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := range dst {
			dst[j] = exactSiLU(a[j]) * bb[j]
		}
	}
}
