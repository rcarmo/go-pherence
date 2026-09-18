//go:build amd64

package simd

// Nonnegative lanes are copied bitwise; mask their polynomial input to zero.
func eluF32VectorLane(x float32) bool { return x >= 0 || eluF32FastPath(x) }

//go:noescape
func eluF32Asm(dst, src []float32)

func eluF32To(dst, src []float32) {
	if !HasVecAsm {
		eluF32ToScalar(dst, src)
		return
	}
	for i := 0; i < len(src); {
		x := src[i]
		if !eluF32VectorLane(x) {
			dst[i] = eluF32Scalar(x)
			i++
			continue
		}
		j := i + 1
		for j < len(src) {
			x = src[j]
			if !eluF32VectorLane(x) {
				break
			}
			j++
		}
		run := j - i
		if run >= 8 {
			n := run &^ 7
			eluF32Asm(dst[i:i+n], src[i:i+n])
			i += n
		}
		for i < j {
			dst[i] = eluF32Scalar(src[i])
			i++
		}
	}
}
