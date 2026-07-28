package nvidia

import (
	"math"
	"testing"
)

func TestSgemmVariantsLiveParity(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	for _, s := range []struct{ m, n, k int }{{1, 1024, 1024}, {4, 1024, 1024}, {8, 1024, 1024}, {32, 1024, 1024}, {227, 1024, 1024}, {32, 3072, 1024}, {1500, 64, 1500}} {
		a := make([]float32, s.m*s.k)
		b := make([]float32, s.k*s.n)
		for i := range a {
			a[i] = float32(i%13-6) / 13
		}
		for i := range b {
			b[i] = float32(i%17-8) / 17
		}
		want := make([]float32, s.m*s.n)
		for i := 0; i < s.m; i++ {
			for j := 0; j < s.n; j++ {
				var v float32
				for p := 0; p < s.k; p++ {
					v += a[i*s.k+p] * b[p*s.n+j]
				}
				want[i*s.n+j] = v
			}
		}
		got, e := SgemmHost(s.m, s.n, s.k, 1, a, b)
		if e != nil {
			t.Fatal(e)
		}
		mx := 0.0
		for i := range got {
			d := math.Abs(float64(got[i] - want[i]))
			if d > mx {
				mx = d
			}
		}
		if mx > 3e-3 {
			t.Fatalf("shape=%+v max=%g", s, mx)
		}
		t.Logf("shape=%+v max=%g", s, mx)
	}
}
