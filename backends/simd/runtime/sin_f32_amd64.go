//go:build amd64

package simd

import "math"

//go:noescape
func sinF32Asm(dst, src []float32)

func sinF32To(dst, src []float32) {
	if !HasVecAsm {
		sinF32ToScalar(dst, src)
		return
	}
	for i := 0; i < len(src); {
		if !sinF32FastPathBits(math.Float32bits(src[i])) {
			dst[i] = sinF32Scalar(src[i])
			i++
			continue
		}
		j := i + 1
		for j < len(src) && sinF32FastPathBits(math.Float32bits(src[j])) {
			j++
		}
		run := j - i
		if run >= 8 {
			n := run &^ 7
			sinF32Asm(dst[i:i+n], src[i:i+n])
			i += n
		}
		for i < j {
			dst[i] = sinF32Scalar(src[i])
			i++
		}
	}
}
