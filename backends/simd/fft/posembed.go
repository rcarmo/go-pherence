package fft

import "math"

// SinusoidalPosEmbed generates sinusoidal position embeddings.
// Returns [maxLen * dModel] flat: PE[pos][2i] = sin, PE[pos][2i+1] = cos.
// Optimized with precomputed frequency table.
func SinusoidalPosEmbed(maxLen, dModel int) []float32 {
	if maxLen <= 0 || dModel <= 0 {
		return nil
	}

	out := make([]float32, maxLen*dModel)

	// Precompute inverse frequency table
	halfDim := dModel / 2
	invFreq := make([]float64, halfDim)
	for i := 0; i < halfDim; i++ {
		invFreq[i] = 1.0 / math.Pow(10000, float64(2*i)/float64(dModel))
	}

	// Generate embeddings
	for pos := 0; pos < maxLen; pos++ {
		off := pos * dModel
		p := float64(pos)
		for i := 0; i < halfDim; i++ {
			angle := p * invFreq[i]
			out[off+2*i] = float32(math.Sin(angle))
			if 2*i+1 < dModel {
				out[off+2*i+1] = float32(math.Cos(angle))
			}
		}
	}

	return out
}

// AddPosEmbed adds positional embeddings to hidden states in-place.
// h: [seqLen * dModel], pe: [maxLen * dModel], startPos: offset into PE table.
func AddPosEmbed(h, pe []float32, seqLen, dModel, startPos int) {
	if len(h) < seqLen*dModel || len(pe) < (startPos+seqLen)*dModel {
		return
	}
	for t := 0; t < seqLen; t++ {
		hOff := t * dModel
		pOff := (startPos + t) * dModel
		for d := 0; d < dModel; d++ {
			h[hOff+d] += pe[pOff+d]
		}
	}
}
