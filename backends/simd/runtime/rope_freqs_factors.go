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
	thetaScale := float32(math.Pow(theta, -2.0/float64(nDims)))
	for pos := 0; pos < maxSeq; pos++ {
		thetaCur := float32(pos)
		for i := 0; i < halfDim; i++ {
			factor := float32(1)
			if i < len(factors) && factors[i] != 0 {
				factor = factors[i]
			}
			angle := thetaCur / factor
			off := (pos*halfDim + i) * 2
			freqs[off] = float32(math.Cos(float64(angle)))
			freqs[off+1] = float32(math.Sin(float64(angle)))
			thetaCur *= thetaScale
		}
	}
	return freqs
}
