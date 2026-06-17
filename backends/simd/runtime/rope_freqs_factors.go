package simd

import "math"

// BuildRoPEFreqsWithFactors builds interleaved cos/sin rotary frequency tables
// matching ggml_rope_ext with optional freq_factors (rope_freqs.weight).
//
// ggml computes angle(pos, i) = pos * theta^(-2*i/nDims) / freq_factors[i]
// for rotary pair i. If factors is nil or too short, factor 1 is used.
func BuildRoPEFreqsWithFactors(maxSeq, halfDim, nDims int, theta float64, factors []float32) []float32 {
	n, ok := CheckedRoPEFreqLen(maxSeq, halfDim)
	if !ok || nDims <= 0 {
		return nil
	}
	if theta <= 0 {
		theta = 10000
	}
	freqs := make([]float32, n)
	for pos := 0; pos < maxSeq; pos++ {
		for i := 0; i < halfDim; i++ {
			factor := float64(1)
			if i < len(factors) && factors[i] != 0 {
				factor = float64(factors[i])
			}
			freq := 1.0 / math.Pow(theta, float64(2*i)/float64(nDims)) / factor
			angle := float64(pos) * freq
			off := (pos*halfDim + i) * 2
			freqs[off] = float32(math.Cos(angle))
			freqs[off+1] = float32(math.Sin(angle))
		}
	}
	return freqs
}
