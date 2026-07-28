package rvv

import (
	"github.com/rcarmo/go-pherence/half"
	"math"
	"testing"
)

func TestPackedOuterTails(t *testing.T) {
	for _, s := range []struct{ m, n, k int }{{1, 1, 8}, {3, 31, 16}, {5, 33, 24}, {7, 65, 32}} {
		a := make([]int8, s.m*s.k)
		b := make([]int8, s.n*s.k)
		for i := range a {
			a[i] = int8(i%7 - 3)
		}
		for i := range b {
			b[i] = int8(i%9 - 4)
		}
		bp := PackB(b, s.n, s.k)
		got := make([]int32, s.m*s.n)
		GemmI8Outer(a, bp, got, s.m, s.n, s.k, 2)
		for i := 0; i < s.m; i++ {
			for j := 0; j < s.n; j++ {
				var w int32
				for p := 0; p < s.k; p++ {
					w += int32(a[i*s.k+p]) * int32(b[j*s.k+p])
				}
				if got[i*s.n+j] != w {
					t.Fatalf("shape=%+v %d,%d got=%d want=%d", s, i, j, got[i*s.n+j], w)
				}
			}
		}
	}
}
func TestF16Outer32Tails(t *testing.T) {
	m, n, k := 5, 35, 17
	a := make([]uint16, m*k)
	b := make([]uint16, n*k)
	for i := range a {
		a[i] = half.F32ToF16(float32(i%7-3) / 7)
	}
	for i := range b {
		b[i] = half.F32ToF16(float32(i%9-4) / 9)
	}
	bp := PackBF16N32(b, n, k)
	got := make([]float32, m*n)
	GemmF16Outer32(a, bp, got, m, n, k, 2)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var w float32
			for p := 0; p < k; p++ {
				w += half.F16ToF32(a[i*k+p]) * half.F16ToF32(b[j*k+p])
			}
			if math.Abs(float64(got[i*n+j]-w)) > 1e-3 {
				t.Fatalf("%d,%d got=%g want=%g", i, j, got[i*n+j], w)
			}
		}
	}
}
