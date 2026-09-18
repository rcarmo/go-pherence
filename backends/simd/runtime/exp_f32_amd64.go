//go:build amd64

package simd

import "math"

//go:noescape
func expF32Asm(dst, src []float32)

func expF32To(dst, src []float32) {
	if !HasVecAsm {
		expF32ToScalar(dst, src)
		return
	}
	for i := 0; i < len(src); {
		if !expF32FastPathBits(math.Float32bits(src[i])) {
			dst[i] = expF32Scalar(src[i])
			i++
			continue
		}
		j := i + 1
		for j < len(src) && expF32FastPathBits(math.Float32bits(src[j])) {
			j++
		}
		run := j - i
		if run >= 8 {
			n := run &^ 7
			expF32Asm(dst[i:i+n], src[i:i+n])
			i += n
		}
		for i < j {
			dst[i] = expF32Scalar(src[i])
			i++
		}
	}
}
