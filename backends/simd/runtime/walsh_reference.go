package simd

import (
	"math"
	"math/bits"
)

// Immutable normalized Sylvester matrix for reference CQ preparation. Do not
// substitute a butterfly here: moving the scale after the additions changes
// FP32 rounding and can cross subsequent activation-quantization thresholds.
var walshReference128 = func() []float32 {
	h := make([]float32, 128*128)
	scale := float32(1 / math.Sqrt(128))
	for row := 0; row < 128; row++ {
		for col := 0; col < 128; col++ {
			v := scale
			if bits.OnesCount(uint(row&col))%2 != 0 {
				v = -v
			}
			h[row*128+col] = v
		}
	}
	return h
}()

// Walsh128ReferenceTo multiplies one group by the normalized dense Hadamard
// matrix using matrix SIMD dispatch. This O(128²) reference operation preserves
// pre-scaled products; it is not the O(128 log 128) packed-inference butterfly.
// It reads/writes the first 128 values and returns false without writing for
// short or overlapping slices. There is no bit-exact cross-ISA guarantee.
func Walsh128ReferenceTo(dst, src []float32) bool {
	if len(dst) < 128 || len(src) < 128 {
		return false
	}
	return MatMul(dst[:128], src[:128], walshReference128, 1, 128, 128, false, false)
}
