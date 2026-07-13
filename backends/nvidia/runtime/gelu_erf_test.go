package nvidia

import (
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"math"
	"testing"
)

func TestDevGELUErfParity(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	x := make([]float32, 20001)
	for i := range x {
		x[i] = -10 + 20*float32(i)/float32(len(x)-1)
	}
	want := append([]float32(nil), x...)
	simd.GELUExact(want, want)
	b := NewDevBufFrom(x)
	defer b.Free()
	DevGELUErf(b, len(x))
	got := b.Data()
	max := 0.0
	idx := 0
	for i := range got {
		d := math.Abs(float64(got[i] - want[i]))
		if d > max {
			max = d
			idx = i
		}
	}
	t.Logf("max=%g x=%g got=%g want=%g", max, x[idx], got[idx], want[idx])
	if max > 3e-6 {
		t.Fatalf("drift")
	}
}
