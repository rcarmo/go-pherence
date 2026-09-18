//go:build amd64

package simd

import "golang.org/x/sys/cpu"

//go:noescape
func x86FeatureInfoECX() uint32

var hasF16CConvert = cpu.X86.HasAVX && x86FeatureInfoECX()&(1<<29) != 0

//go:noescape
func f16LittleEndianToF32Asm(dst []float32, src []byte) int

func f16LittleEndianToF32(dst []float32, src []byte) {
	if len(dst) == 0 {
		return
	}
	if !hasF16CConvert || len(dst) < 8 {
		f16LittleEndianToF32Scalar(dst, src)
		return
	}

	// F16C matches the scalar path for finite values, zeros, subnormals, and
	// infinities, but hardware quiets NaNs. The assembly loop vectorizes until it
	// reaches an 8-lane block containing a NaN, returns the count already
	// processed, and leaves that exceptional block for the scalar converter so we
	// preserve exact NaN payload/signaling bits.
	n := len(dst) &^ 7
	i := 0
	for i < n {
		i += f16LittleEndianToF32Asm(dst[i:n], src[i*2:n*2])
		if i == n {
			break
		}
		f16LittleEndianToF32Scalar(dst[i:i+8], src[i*2:i*2+16])
		i += 8
	}
	if i != len(dst) {
		f16LittleEndianToF32Scalar(dst[i:], src[i*2:])
	}
}
