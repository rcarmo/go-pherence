package omnivoice

import (
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"testing"
)

// Compare projection layouts before changing the production weight contract.
func BenchmarkProjectionLayout(b *testing.B) {
	const rows, in, out = 128, 1024, 3072
	a := make([]float32, rows*in)
	w := make([]float32, out*in)
	wt := make([]float32, in*out)
	c := make([]float32, rows*out)
	for i := range a {
		a[i] = float32(i%71) * .01
	}
	for i := range w {
		w[i] = float32(i%13) * .01
	}
	for o := 0; o < out; o++ {
		for k := 0; k < in; k++ {
			wt[k*out+o] = w[o*in+k]
		}
	}
	scratch := make([]float32, in*16)
	b.Run("Packed", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			clear(c)
			simd.SgemmNTPackedTo(c, a, w, scratch, rows, out, in, 1, in, in, out)
		}
	})
	b.Run("NT", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			clear(c)
			simd.SgemmNTTo(c, a, w, rows, out, in, 1, in, in, out)
		}
	})
	b.Run("NN_pretransposed", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			clear(c)
			simd.SgemmNNTo(c, a, wt, rows, out, in, 1, in, out, out)
		}
	})
}
