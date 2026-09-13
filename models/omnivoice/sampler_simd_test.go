package omnivoice

import (
	"math"
	"math/rand"
	"testing"
)

func TestLogSoftmaxSIMDParity(t *testing.T) {
	rng := rand.New(rand.NewSource(82))
	for _, n := range []int{1, 7, 8, 9, 16, 1025} {
		for _, scale := range []float32{0.1, 1, 10, 100, 10000} {
			src, want, tmp := make([]float32, n), make([]float32, n), make([]float32, n)
			for i := range src {
				src[i] = float32(rng.NormFloat64()) * scale
			}
			copy(want, src)
			logSoftmaxInPlace(want)
			logSoftmaxSIMDInPlace(src, tmp)
			for i, v := range src {
				d := math.Abs(float64(v) - float64(want[i]))
				if d > 4e-6 && d > 2e-6*math.Abs(float64(want[i])) {
					t.Fatalf("n%d scale%g i%d got%g want%g", n, scale, i, v, want[i])
				}
			}
		}
	}
	for _, src := range [][]float32{{float32(math.Inf(-1)), float32(math.Inf(-1))}, {float32(math.Inf(1)), 2, float32(math.Inf(1))}, {float32(math.NaN()), 1, 2}, {1, float32(math.NaN()), float32(math.Inf(1))}, {1, float32(math.Inf(-1)), 2}} {
		want := append([]float32(nil), src...)
		logSoftmaxInPlace(want)
		logSoftmaxSIMDInPlace(src, make([]float32, len(src)))
		for i, v := range src {
			if math.IsNaN(float64(want[i])) {
				if !math.IsNaN(float64(v)) {
					t.Fatal("NaN mismatch")
				}
			} else if v != want[i] {
				t.Fatalf("exceptional got%v want%v", src, want)
			}
		}
	}
}

func BenchmarkLogSoftmaxSIMD(b *testing.B) {
	src, x, tmp := make([]float32, 1025), make([]float32, 1025), make([]float32, 1025)
	for i := range src {
		src[i] = float32(i%137-68) / 8
	}
	b.Run("scalar", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			copy(x, src)
			logSoftmaxInPlace(x)
		}
	})
	b.Run("simd", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			copy(x, src)
			logSoftmaxSIMDInPlace(x, tmp)
		}
	})
}

func TestLogSoftmaxSIMDExtremeFinite(t *testing.T) {
	for _, src := range [][]float32{{math.MaxFloat32, -math.MaxFloat32, 0}, {-math.MaxFloat32, -math.MaxFloat32}, {0, math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32}} {
		want := append([]float32(nil), src...)
		logSoftmaxInPlace(want)
		logSoftmaxSIMDInPlace(src, make([]float32, len(src)))
		for i := range src {
			if src[i] != want[i] {
				t.Fatalf("got%v want%v", src, want)
			}
		}
	}
}
