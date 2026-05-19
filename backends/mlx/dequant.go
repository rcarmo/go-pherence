package mlx

// Dequant dequantizes an MLX affine quantized weight to F32.
// weight[outDim, inDim/packFactor] × scales[outDim, numGroups] + biases[outDim, numGroups]
// Returns [outDim, inDim] float32.
func Dequant(qw *QuantWeight) []float32 {
	if err := ValidateQuantWeight(qw); err != nil {
		return nil
	}
	outLen, ok := checkedMulInt(qw.OutDim, qw.InDim)
	if !ok {
		return nil
	}
	out := make([]float32, outLen)
	packFactor := 32 / qw.Bits
	mask := uint32((1 << qw.Bits) - 1)

	for row := 0; row < qw.OutDim; row++ {
		rowOff := row * qw.InDim
		packedOff := row * (qw.InDim / packFactor)
		scaleOff := row * qw.Groups

		for g := 0; g < qw.Groups; g++ {
			scale := qw.Scales[scaleOff+g]
			bias := qw.Biases[scaleOff+g]
			gStart := g * qw.GroupSize

			for e := 0; e < qw.GroupSize; e++ {
				idx := gStart + e
				packIdx := idx / packFactor
				bitPos := uint(idx%packFactor) * uint(qw.Bits)
				val := (qw.Weight[packedOff+packIdx] >> bitPos) & mask
				out[rowOff+idx] = float32(val)*scale + bias
			}
		}
	}
	return out
}
