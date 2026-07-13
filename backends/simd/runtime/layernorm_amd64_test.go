//go:build amd64

package simd

import (
	"math"
	"math/rand"
	"strconv"
	"testing"
)

func TestLayerNormAffineAVX2AgainstScalar(t *testing.T) {
	if !HasVecAsm {
		t.Skip("AVX2/FMA unavailable")
	}
	for _, cols := range []int{8, 9, 15, 16, 17, 31, 32, 33, 1023, 1024, 4096} {
		t.Run(strconv.Itoa(cols), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(cols)))
			x := make([]float32, cols)
			gamma := make([]float32, cols)
			beta := make([]float32, cols)
			for i := range x {
				x[i] = (rng.Float32()*2 - 1) * 12
				gamma[i] = rng.Float32()*2 - 1
				beta[i] = rng.Float32()*2 - 1
			}
			want := make([]float32, cols)
			got := make([]float32, cols)
			layerNormAffineRowGo(want, x, gamma, beta, 1e-6)
			layerNormAffineRowTo(got, x, gamma, beta, 1e-6)
			assertCloseLayerNorm(t, got, want, 3e-5)

			alias := append([]float32(nil), x...)
			layerNormAffineRowTo(alias, alias, gamma, beta, 1e-6)
			assertCloseLayerNorm(t, alias, want, 3e-5)
		})
	}
}

func TestLayerNormAffineAVX2ConstantAndTail(t *testing.T) {
	if !HasVecAsm {
		t.Skip("AVX2/FMA unavailable")
	}
	for _, cols := range []int{8, 13, 24, 29} {
		x := make([]float32, cols)
		gamma := make([]float32, cols)
		beta := make([]float32, cols)
		for i := range x {
			x[i], gamma[i], beta[i] = 7, 0.5, float32(i)-3
		}
		got := make([]float32, cols)
		layerNormAffineRowTo(got, x, gamma, beta, 1e-6)
		assertCloseLayerNorm(t, got, beta, 1e-6)
	}
}

func BenchmarkLayerNormAffine1024(b *testing.B) {
	x := make([]float32, 1024)
	gamma := make([]float32, 1024)
	beta := make([]float32, 1024)
	out := make([]float32, 1024)
	for i := range x {
		x[i], gamma[i] = float32(i%31)-15, 1
	}
	bench := func(b *testing.B, fn func([]float32, []float32, []float32, []float32, float32)) {
		b.SetBytes(int64(len(x) * 4 * 4))
		for range b.N {
			fn(out, x, gamma, beta, 1e-6)
		}
	}
	b.Run("scalar", func(b *testing.B) { bench(b, layerNormAffineRowGo) })
	b.Run("dispatch", func(b *testing.B) { bench(b, layerNormAffineRowTo) })
}

func assertCloseLayerNorm(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > tol || math.IsNaN(d) {
			t.Fatalf("index %d: got %.9g want %.9g diff %.3g > %.3g", i, got[i], want[i], d, tol)
		}
	}
}
