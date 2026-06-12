package rvv

import (
	"math"
	"testing"
)

func TestFastExpAccuracy(t *testing.T) {
	maxRel := 0.0
	for x := float32(-10); x <= 10; x += 0.01 {
		got := float64(FastExp(x))
		want := math.Exp(float64(x))
		if want == 0 {
			continue
		}
		rel := math.Abs(got-want) / want
		if rel > maxRel {
			maxRel = rel
		}
	}
	t.Logf("maxRelErr=%.4f%%", maxRel*100)
	if maxRel > 0.10 {
		t.Fatalf("too inaccurate: maxRelErr=%.4f%%", maxRel*100)
	}
}

func BenchmarkFastExp(b *testing.B) {
	x := make([]float32, 4096)
	for i := range x {
		x[i] = float32(i%1000-500) * 0.01
	}
	b.SetBytes(int64(len(x) * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := range x {
			x[j] = FastExp(x[j])
		}
	}
}

func BenchmarkMathExp(b *testing.B) {
	x := make([]float32, 4096)
	for i := range x {
		x[i] = float32(i%1000-500) * 0.01
	}
	b.SetBytes(int64(len(x) * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := range x {
			x[j] = float32(math.Exp(float64(x[j])))
		}
	}
}
