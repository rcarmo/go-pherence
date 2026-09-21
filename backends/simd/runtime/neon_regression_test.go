package simd

import (
	"math"
	"testing"
)

// Native ARM qualification found a scalar/vector register alias in NT reduction
// and RMSNorm tails, plus XTN encoded at the wrong element width. Cover every
// lane/tail boundary, not just lengths divisible by the vector width.
func TestSGEMMNTAllReductionLanesAndTails(t *testing.T) {
	for k := 1; k <= 65; k++ {
		lda, ldb, ldc := k+3, k+5, 6
		a, b, c := make([]float32, 2*lda), make([]float32, 3*ldb), make([]float32, 2*ldc)
		for i := range a {
			a[i] = float32(i%13-6) / 7
		}
		for i := range b {
			b[i] = float32(i%17-8) / 9
		}
		for i := range c {
			c[i] = .25
		}
		if !SgemmNTTo(c, a, b, 2, 3, k, .7, lda, ldb, ldc) {
			t.Fatal("rejected")
		}
		for i := 0; i < 2; i++ {
			for j := 0; j < 3; j++ {
				want := float64(.25)
				for p := 0; p < k; p++ {
					want += .7 * float64(a[i*lda+p]) * float64(b[j*ldb+p])
				}
				if math.Abs(float64(c[i*ldc+j])-want) > 2e-5 {
					t.Fatalf("k=%d [%d,%d] got %g want %g", k, i, j, c[i*ldc+j], want)
				}
			}
			for j := 3; j < ldc; j++ {
				if c[i*ldc+j] != .25 {
					t.Fatal("overwrote C tail")
				}
			}
		}
	}
}
func TestRMSNormEveryTail(t *testing.T) {
	for n := 1; n <= 65; n++ {
		x, w := make([]float32, n), make([]float32, n)
		for i := range x {
			x[i] = float32(i%11-5) * .13
			w[i] = float32(i%5+1) * .27
		}
		want := append([]float32(nil), x...)
		rmsNormGo(want, w, 1e-5)
		RMSNorm(x, w, 1e-5)
		for i := range x {
			if math.Abs(float64(x[i]-want[i])) > 2e-5 {
				t.Fatalf("n=%d i=%d got%g want%g", n, i, x[i], want[i])
			}
		}
	}
}
func TestBF16NarrowAndAddEveryTail(t *testing.T) {
	for n := 1; n <= 33; n++ {
		a, b := make([]uint16, n), make([]uint16, n)
		f := make([]float32, n)
		for i := range a {
			f[i] = float32(i%13-6) * .25
			a[i] = uint16(math.Float32bits(f[i]) >> 16)
			b[i] = uint16(math.Float32bits(float32(i%7+1)) >> 16)
		}
		out := make([]uint16, n+2)
		out[n], out[n+1] = 0x55aa, 0x55aa
		BF16NarrowFromF32(out[:n], f)
		for i := 0; i < n; i++ {
			if out[i] != a[i] {
				t.Fatalf("narrow n=%d i=%d got%x want%x", n, i, out[i], a[i])
			}
		}
		BF16VecAddAsm(out[:n], a, b)
		for i := 0; i < n; i++ {
			sum := math.Float32frombits(uint32(a[i])<<16) + math.Float32frombits(uint32(b[i])<<16)
			want := uint16(math.Float32bits(sum) >> 16)
			if out[i] != want {
				t.Fatalf("add n=%d i=%d got%x want%x", n, i, out[i], want)
			}
		}
		if out[n] != 0x55aa || out[n+1] != 0x55aa {
			t.Fatal("BF16 tail overwritten")
		}
	}
}
