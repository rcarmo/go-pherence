package quant

import simdq4 "github.com/rcarmo/go-pherence/backends/simd/runtime/q4"

// DequantGPTQ dequantizes a GPTQ INT4 weight tensor to float32.
func DequantGPTQ(qweight, qzeros, gIdx []int32, scales []float32, inFeatures, outFeatures int, sym bool) []float32 {
	return simdq4.Dequant(qweight, qzeros, gIdx, scales, inFeatures, outFeatures, sym)
}

// DequantGPTQTo dequantizes a GPTQ INT4 weight tensor into caller-owned storage.
func DequantGPTQTo(out []float32, qweight, qzeros, gIdx []int32, scales []float32, inFeatures, outFeatures int, sym bool) bool {
	return simdq4.DequantTo(out, qweight, qzeros, gIdx, scales, inFeatures, outFeatures, sym)
}

// DequantGPTQSym is an optimized parallel symmetric dequantization (zero point = 8).
func DequantGPTQSym(qweight, gIdx []int32, scales []float32, inFeatures, outFeatures int) []float32 {
	return simdq4.DequantSym(qweight, gIdx, scales, inFeatures, outFeatures)
}

// DequantGPTQSymTo dequantizes a symmetric GPTQ tensor into caller-owned storage.
func DequantGPTQSymTo(out []float32, qweight, gIdx []int32, scales []float32, inFeatures, outFeatures int) bool {
	return simdq4.DequantSymTo(out, qweight, gIdx, scales, inFeatures, outFeatures)
}

// Float16ToFloat32 converts a uint16 IEEE 754 half-precision to float32.
func Float16ToFloat32(h uint16) float32 { return simdq4.Float16ToFloat32(h) }
