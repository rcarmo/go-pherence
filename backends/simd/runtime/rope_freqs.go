package simd

import (
	"github.com/rcarmo/go-pherence/internal/checked"
	"math"
)

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
	thetaScale := float32(math.Pow(theta, -2.0/float64(headDim)))
	for pos := 0; pos < maxSeq; pos++ {
		thetaCur := float32(pos)
		for i := 0; i < halfDim; i++ {
			off := (pos*halfDim + i) * 2
			freqs[off] = float32(math.Cos(float64(thetaCur)))
			freqs[off+1] = float32(math.Sin(float64(thetaCur)))
			thetaCur *= thetaScale
		}
	}
	return freqs
}

// CheckedRoPEFreqLen returns the interleaved cos/sin table length.
func CheckedRoPEFreqLen(maxSeq, halfDim int) (int, bool) {
	if maxSeq <= 0 || halfDim <= 0 {
		return 0, false
	}
	pairs, okPairs := checked.MulInt(maxSeq, halfDim)
	n, okN := checked.MulInt(pairs, 2)
	if !okPairs || !okN {
		return 0, false
	}
	return n, true
}
