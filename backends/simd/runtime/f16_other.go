//go:build !amd64

package simd

var hasF16CConvert = false

func f16LittleEndianToF32(dst []float32, src []byte) {
	f16LittleEndianToF32Scalar(dst, src)
}
