package quant

import simdq4 "github.com/rcarmo/go-pherence/backends/simd/runtime/q4"

// ValidateGPTQ checks GPTQ tensor lengths and dimensions before dequantization.
func ValidateGPTQ(qweight, qzeros, gIdx []int32, scales []float32, inFeatures, outFeatures int, sym bool) error {
	return simdq4.Validate(qweight, qzeros, gIdx, scales, inFeatures, outFeatures, sym)
}

// ValidateGPTQSym checks symmetric GPTQ tensor lengths and dimensions.
func ValidateGPTQSym(qweight, gIdx []int32, scales []float32, inFeatures, outFeatures int) error {
	return simdq4.ValidateSym(qweight, gIdx, scales, inFeatures, outFeatures)
}

func checkedMulInt(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if b != 0 && a > maxInt/b {
		return 0, false
	}
	return a * b, true
}
