package q4

func GemvSym(out, x []float32, qweight, gIdx []int32, scales []float32, inDim, outDim int) {
	if err := ValidateGemvSym(out, x, qweight, gIdx, scales, inDim, outDim); err != nil {
		return
	}
	for j := 0; j < outDim; j++ {
		out[j] = 0
	}
	for packIdx := 0; packIdx < inDim/8; packIdx++ {
		qrow := qweight[packIdx*outDim : (packIdx+1)*outDim]
		for bit := 0; bit < 8; bit++ {
			i := packIdx*8 + bit
			g := int(gIdx[i])
			scaleRow := scales[g*outDim : (g+1)*outDim]
			xi := x[i]
			shift := uint(bit) * 4
			for j := 0; j < outDim; j++ {
				qw := (qrow[j] >> shift) & 0xF
				out[j] += xi * scaleRow[j] * float32(qw-8)
			}
		}
	}
}
