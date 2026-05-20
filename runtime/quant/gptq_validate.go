package quant

import simdq4 "github.com/rcarmo/go-pherence/backends/simd/quant/q4"

// ValidateGPTQ checks GPTQ tensor lengths and dimensions before dequantization.
func ValidateGPTQ(qweight, qzeros, gIdx []int32, scales []float32, inFeatures, outFeatures int, sym bool) error {
	return simdq4.Validate(qweight, qzeros, gIdx, scales, inFeatures, outFeatures, sym)
}

// ValidateGPTQSym checks symmetric GPTQ tensor lengths and dimensions.
func ValidateGPTQSym(qweight, gIdx []int32, scales []float32, inFeatures, outFeatures int) error {
	return simdq4.ValidateSym(qweight, gIdx, scales, inFeatures, outFeatures)
}
