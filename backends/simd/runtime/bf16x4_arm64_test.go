//go:build arm64

package simd

import (
	"math"
	"math/rand"
	"testing"
)

// Preserve four independent scalar row-dot bits on ARM64. A shared-activation
// single-pass BF16 trial differed at cols=7 and changed pinned 25-frame PCM;
// a numerical-tolerance-only gate would not have caught that regression.
func TestARM64BF16F32x4BitwiseScalar(t *testing.T) {
	for _, cols := range []int{1, 7, 8, 9, 16, 17, 48, 256, 768, 1024, 4096} {
		for seed := int64(0); seed < 16; seed++ {
			rng := rand.New(rand.NewSource(seed*100000 + int64(cols)))
			w, x := make([]uint16, 4*cols), make([]float32, cols)
			for i := range w {
				w[i] = F32ToBF16((rng.Float32()*2 - 1) * 0.5)
			}
			for i := range x {
				x[i] = (rng.Float32()*2 - 1) * 0.5
			}
			a, b, c, d := bf16DotF32x4(w, x, cols)
			got := [4]float32{a, b, c, d}
			for row := 0; row < 4; row++ {
				want := BF16DotF32(w[row*cols:(row+1)*cols], x)
				if math.Float32bits(got[row]) != math.Float32bits(want) {
					t.Fatalf("cols=%d seed=%d row=%d got=%g (%08x) scalar=%g (%08x)", cols, seed, row, got[row], math.Float32bits(got[row]), want, math.Float32bits(want))
				}
			}
		}
	}
}
