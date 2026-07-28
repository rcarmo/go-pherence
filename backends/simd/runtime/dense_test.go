package simd

import (
	"math"
	"testing"
)

func TestDenseDispatchParity(t *testing.T) {
	for _, s := range []struct{ m, n, k int }{{1, 63, 65}, {2, 64, 64}, {5, 257, 67}, {32, 1024, 1024}, {257, 64, 64}} {
		a := randFloats(s.m*s.k, int64(s.m))
		nt := randFloats(s.n*s.k, int64(s.n))
		nn := make([]float32, s.k*s.n)
		for p := 0; p < s.k; p++ {
			for j := 0; j < s.n; j++ {
				nn[p*s.n+j] = nt[j*s.k+p]
			}
		}
		for _, kind := range []string{"nt", "nn"} {
			want := make([]float32, s.m*s.n)
			got := make([]float32, len(want))
			var ok1, ok2 bool
			if kind == "nt" {
				ok1 = SgemmNTTo(want, a, nt, s.m, s.n, s.k, 1, s.k, s.k, s.n)
				ok2 = DenseNTTo(got, a, nt, s.m, s.n, s.k, 1, s.k, s.k, s.n)
			} else {
				ok1 = SgemmNNTo(want, a, nn, s.m, s.n, s.k, 1, s.k, s.n, s.n)
				ok2 = DenseNNTo(got, a, nn, s.m, s.n, s.k, 1, s.k, s.n, s.n)
			}
			if !ok1 || !ok2 {
				t.Fatalf("rejected %s %+v", kind, s)
			}
			for i := range want {
				if d := math.Abs(float64(want[i] - got[i])); d > 3e-5 {
					t.Fatalf("%s %+v [%d] diff=%g", kind, s, i, d)
				}
			}
		}
	}
}
