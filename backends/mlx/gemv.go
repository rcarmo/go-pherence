package mlx

// Gemv performs matrix-vector multiply with MLX quantized weight.
// out[outDim] = W_mlx[outDim, inDim] · x[inDim] (dequantized on-the-fly)
func Gemv(out, x []float32, qw *QuantWeight) { _ = GemvTo(out, x, qw) }

// GemvTo performs MLX quantized GEMV into caller-owned output and reports
// malformed inputs while preserving Gemv's no-allocation behavior.
func GemvTo(out, x []float32, qw *QuantWeight) bool {
	if err := ValidateQuantWeight(qw); err != nil || len(out) < qw.OutDim || len(x) < qw.InDim {
		return false
	}
	if qw.Bits == 4 && qw.GroupSize%8 == 0 {
		// Dispatch hook kept explicit so AVX2/NEON MLX4 kernels can be wired
		// without changing callers. Scalar remains active until hasGemv4Asm flips.
		gemv4Scalar(out, x, qw)
		return true
	}
	gemvScalar(out, x, qw)
	return true
}

func gemvScalar(out, x []float32, qw *QuantWeight) {
	packFactor := 32 / qw.Bits
	mask := uint32((1 << qw.Bits) - 1)

	for row := 0; row < qw.OutDim; row++ {
		packedOff := row * (qw.InDim / packFactor)
		scaleOff := row * qw.Groups
		sum := float32(0)

		for g := 0; g < qw.Groups; g++ {
			scale := qw.Scales[scaleOff+g]
			bias := qw.Biases[scaleOff+g]
			gStart := g * qw.GroupSize

			gsum := float32(0)
			xsum := float32(0) // for bias accumulation

			for e := 0; e < qw.GroupSize; e++ {
				idx := gStart + e
				packIdx := idx / packFactor
				bitPos := uint(idx%packFactor) * uint(qw.Bits)
				val := float32((qw.Weight[packedOff+packIdx] >> bitPos) & mask)
				gsum += val * x[idx]
				xsum += x[idx]
			}
			sum += gsum*scale + xsum*bias
		}
		out[row] = sum
	}
}

func gemv4Scalar(out, x []float32, qw *QuantWeight) {
	packedPerRow := qw.InDim / 8
	packsPerGroup := qw.GroupSize / 8
	var groupXSumsStack [256]float32
	groupXSums := groupXSumsStack[:]
	if qw.Groups > len(groupXSumsStack) {
		groupXSums = make([]float32, qw.Groups)
	} else {
		groupXSums = groupXSums[:qw.Groups]
	}
	for g := 0; g < qw.Groups; g++ {
		xBase := g * qw.GroupSize
		xsum := float32(0)
		for p := 0; p < packsPerGroup; p++ {
			xi := xBase + p*8
			xsum += x[xi] + x[xi+1] + x[xi+2] + x[xi+3] + x[xi+4] + x[xi+5] + x[xi+6] + x[xi+7]
		}
		groupXSums[g] = xsum
	}
	for row := 0; row < qw.OutDim; row++ {
		packedOff := row * packedPerRow
		scaleOff := row * qw.Groups
		sum := float32(0)
		for g := 0; g < qw.Groups; g++ {
			scale := qw.Scales[scaleOff+g]
			bias := qw.Biases[scaleOff+g]
			packBase := packedOff + g*packsPerGroup
			xBase := g * qw.GroupSize
			gsum := float32(0)
			for p := 0; p < packsPerGroup; p++ {
				packed := qw.Weight[packBase+p]
				xi := xBase + p*8
				x0 := x[xi]
				x1 := x[xi+1]
				x2 := x[xi+2]
				x3 := x[xi+3]
				x4 := x[xi+4]
				x5 := x[xi+5]
				x6 := x[xi+6]
				x7 := x[xi+7]
				gsum += float32(packed&0xF)*x0 + float32((packed>>4)&0xF)*x1 + float32((packed>>8)&0xF)*x2 + float32((packed>>12)&0xF)*x3 +
					float32((packed>>16)&0xF)*x4 + float32((packed>>20)&0xF)*x5 + float32((packed>>24)&0xF)*x6 + float32((packed>>28)&0xF)*x7
			}
			sum += gsum*scale + groupXSums[g]*bias
		}
		out[row] = sum
	}
}
