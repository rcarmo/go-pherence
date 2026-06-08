package rvv

import (
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestF32ToF16RVV(t *testing.T) {
	vals := []float32{0, 1, -1, 0.5, 2, -3, 65504, 1.0 / 3.0}
	for n := 1; n <= 257; n++ {
		src := make([]float32, n)
		dst := make([]uint16, n)
		for i := range src {
			src[i] = vals[i%len(vals)]
		}
		F32ToF16RVV(dst, src)
		for i := range src {
			want := half.F32ToF16(src[i])
			if dst[i] != want {
				t.Fatalf("n=%d i=%d got 0x%04x want 0x%04x", n, i, dst[i], want)
			}
		}
	}
}

func BenchmarkF32ToF16RVV_1500x64(b *testing.B) {
	src := make([]float32, 1500*64)
	dst := make([]uint16, len(src))
	for i := range src {
		src[i] = float32(i%17) / 7
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		F32ToF16RVV(dst, src)
	}
}
