package quant

import simdnvfp4 "github.com/rcarmo/go-pherence/backends/simd/quant/nvfp4"

type NVFP4Weight = simdnvfp4.NVFP4Weight

func ValidateNVFP4Weight(qw *NVFP4Weight) error          { return simdnvfp4.ValidateNVFP4Weight(qw) }
func DecodeFP4E2M1(code byte) float32                    { return simdnvfp4.DecodeFP4E2M1(code) }
func DecodeF8E4M3(code byte) float32                     { return simdnvfp4.DecodeF8E4M3(code) }
func UnpackNVFP4(packed []byte, count int) []float32     { return simdnvfp4.UnpackNVFP4(packed, count) }
func DequantNVFP4(qw *NVFP4Weight) []float32             { return simdnvfp4.DequantNVFP4(qw) }
func DequantNVFP4To(out []float32, qw *NVFP4Weight) bool { return simdnvfp4.DequantNVFP4To(out, qw) }
func GemvNVFP4(out, x []float32, qw *NVFP4Weight)        { simdnvfp4.GemvNVFP4(out, x, qw) }
func GemvNVFP4To(out, x []float32, qw *NVFP4Weight) bool { return simdnvfp4.GemvNVFP4To(out, x, qw) }
func GemmNVFP4(out, x []float32, batch int, qw *NVFP4Weight) bool {
	return simdnvfp4.GemmNVFP4(out, x, batch, qw)
}
func GemvNVFP4Reference(out, x []float32, qw *NVFP4Weight) bool {
	return simdnvfp4.GemvNVFP4Reference(out, x, qw)
}
