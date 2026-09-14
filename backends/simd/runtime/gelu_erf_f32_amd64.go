//go:build amd64

package simd

import "math"

//go:noescape
func geluErfF32Asm(dst, src []float32)

func geluErfF32To(dst, src []float32) {
	if !HasVecAsm {
		geluErfF32ToScalar(dst, src)
		return
	}
	for i := 0; i < len(src); {
		if !geluErfF32FastPathBits(math.Float32bits(src[i])) {
			dst[i] = geluErfF32Scalar(src[i])
			i++
			continue
		}
		j := i + 1
		for j < len(src) && geluErfF32FastPathBits(math.Float32bits(src[j])) {
			j++
		}
		run := j - i
		if run >= 8 {
			n := run &^ 7
			geluErfF32Asm(dst[i:i+n], src[i:i+n])
			i += n
		}
		for i < j {
			dst[i] = geluErfF32Scalar(src[i])
			i++
		}
	}
}
