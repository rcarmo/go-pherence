package simd

import (
	"encoding/binary"

	"github.com/rcarmo/go-pherence/half"
)

// F16LittleEndianToF32 decodes little-endian IEEE-754 half-precision bytes into
// dst. It returns false and leaves dst unchanged when src is not exactly
// 2*len(dst) bytes.
func F16LittleEndianToF32(dst []float32, src []byte) bool {
	if len(src) != len(dst)*2 {
		return false
	}
	f16LittleEndianToF32(dst, src)
	return true
}

func f16LittleEndianToF32Scalar(dst []float32, src []byte) {
	for i := range dst {
		dst[i] = half.F16ToF32(binary.LittleEndian.Uint16(src[i*2:]))
	}
}
