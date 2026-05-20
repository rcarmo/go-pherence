package rope

import "math"

// BuildRoPEFreqs builds interleaved cos/sin rotary frequency tables for
// maxSeq positions and halfDim rotary pairs. It is the SIMD-owned reference
// for model RoPE table generation.
func BuildRoPEFreqs(maxSeq, halfDim, headDim int, theta float64) []float32 {
	n, ok := CheckedRoPEFreqLen(maxSeq, halfDim)
	if !ok || headDim <= 0 {
		return nil
	}
	if theta <= 0 {
		theta = 10000
	}
	freqs := make([]float32, n)
	for pos := 0; pos < maxSeq; pos++ {
		for i := 0; i < halfDim; i++ {
			freq := 1.0 / math.Pow(theta, float64(2*i)/float64(headDim))
			angle := float64(pos) * freq
			off := (pos*halfDim + i) * 2
			freqs[off] = float32(math.Cos(angle))
			freqs[off+1] = float32(math.Sin(angle))
		}
	}
	return freqs
}

// CheckedRoPEFreqLen returns the interleaved cos/sin table length.
func CheckedRoPEFreqLen(maxSeq, halfDim int) (int, bool) {
	if maxSeq <= 0 || halfDim <= 0 {
		return 0, false
	}
	pairs, okPairs := checkedMulInt(maxSeq, halfDim)
	n, okN := checkedMulInt(pairs, 2)
	if !okPairs || !okN {
		return 0, false
	}
	return n, true
}
