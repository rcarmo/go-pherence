package norm

import "math"

func LayerNormLastAxis(data []float32, shape []int, gamma, beta []float32, eps float32) []float32 {
	if len(shape) == 0 {
		panic("layernorm: scalar tensor")
	}
	lastDim := shape[len(shape)-1]
	if lastDim <= 0 {
		return nil
	}
	total := shapeSizeChecked("layernorm", shape)
	if total < 0 || len(data) < total {
		panic("layernorm: invalid backing data")
	}
	if (gamma == nil) != (beta == nil) {
		panic("layernorm: gamma and beta must both be present or nil")
	}
	if gamma != nil && (len(gamma) < lastDim || len(beta) < lastDim) {
		panic("layernorm: gamma/beta length mismatch")
	}
	outerSize := total / lastDim
	out := make([]float32, total)
	for i := 0; i < outerSize; i++ {
		off := i * lastDim
		row := data[off : off+lastDim]

		mean := float32(0)
		for _, v := range row {
			mean += v
		}
		mean /= float32(lastDim)

		variance := float32(0)
		for _, v := range row {
			d := v - mean
			variance += d * d
		}
		variance /= float32(lastDim)
		stdInv := float32(1.0 / math.Sqrt(float64(variance+eps)))

		for j := 0; j < lastDim; j++ {
			v := (row[j] - mean) * stdInv
			if gamma != nil {
				v = gamma[j]*v + beta[j]
			}
			out[off+j] = v
		}
	}
	return out
}
