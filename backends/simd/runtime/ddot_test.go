package simd

import (
	"math"
	"math/rand"
	"testing"
)

func TestDdotAgainstScalar(t *testing.T) {
	for _, n := range []int{0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 400, 401} {
		rng := rand.New(rand.NewSource(int64(n + 1)))
		x := make([]float64, n)
		y := make([]float64, n)
		for i := range x {
			x[i] = (rng.Float64()*2 - 1) * 10
			y[i] = rng.Float64()*2 - 1
		}
		want := ddotGo(x, y)
		got := Ddot(x, y)
		tol := 2e-13 * math.Max(1, math.Abs(want))
		if math.Abs(got-want) > tol {
			t.Fatalf("n=%d Ddot=%g scalar=%g diff=%g tolerance=%g", n, got, want, math.Abs(got-want), tol)
		}
	}
}

func TestDdotCommonPrefix(t *testing.T) {
	if got := Ddot([]float64{1, 2, 3}, []float64{4, 5}); got != 14 {
		t.Fatalf("Ddot mismatched lengths=%v want 14", got)
	}
}

func BenchmarkDdot400(b *testing.B) {
	x := make([]float64, 400)
	y := make([]float64, 400)
	for i := range x {
		x[i], y[i] = float64(i%17)-8, float64(i%13)-6
	}
	b.SetBytes(int64(len(x) * 16))
	for range b.N {
		_ = Ddot(x, y)
	}
}
