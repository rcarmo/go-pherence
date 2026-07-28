package simd

import (
	"math"
	"testing"
	"unsafe"
)

func TestSgemmNTBlockedFMAFourColumnAndTails(t *testing.T) {
	for _, shape := range []struct{ m, n, k int }{
		{1, 4, 8}, {2, 5, 9}, {3, 6, 15}, {4, 7, 16}, {5, 9, 17},
		{3, 11, 31}, {5, 13, 32}, {7, 15, 33}, {9, 65, 67},
	} {
		s := shape
		a := randFloats(s.m*s.k, int64(100+s.m))
		b := randFloats(s.n*s.k, int64(200+s.n))
		want := randFloats(s.m*s.n, int64(300+s.k))
		got := append([]float32(nil), want...)
		if !SgemmNTTo(want, a, b, s.m, s.n, s.k, 0.75, s.k, s.k, s.n) {
			t.Fatalf("serial rejected shape %+v", s)
		}
		SgemmNTBlockedFMA(s.m, s.n, s.k, 0.75, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&got[0]), s.k, s.k, s.n)
		for i := range want {
			if diff := math.Abs(float64(got[i] - want[i])); diff > 3e-5 {
				t.Fatalf("shape=%+v index=%d got=%g want=%g diff=%g", s, i, got[i], want[i], diff)
			}
		}
	}
}
