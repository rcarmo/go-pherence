package q4

func GemvSym(out, x []float32, qweight, gIdx []int32, scales []float32, inDim, outDim int) {
	if err := ValidateGemvSym(out, x, qweight, gIdx, scales, inDim, outDim); err != nil {
		return
	}
	for j := 0; j < outDim; j++ {
		var sum float32
		for packIdx := 0; packIdx < inDim/8; packIdx++ {
			packed := qweight[packIdx*outDim+j]
			for bit := 0; bit < 8; bit++ {
				i := packIdx*8 + bit
				qw := (packed >> (uint(bit) * 4)) & 0xF
				g := int(gIdx[i])
				scale := scales[g*outDim+j]
				sum += x[i] * scale * float32(qw-8)
			}
		}
		out[j] = sum
	}
}
