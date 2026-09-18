package gliner2

import (
	"fmt"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"math"
)

// IntervalPrefixScore implements upstream's half-open prefix evidence. Indices
// are clamped to [0,len(prefix)-1]; mean correction uses the clamped length.
func IntervalPrefixScore(prefix []float32, start, end int, mean *float32) (float32, error) {
	if len(prefix) == 0 {
		return 0, fmt.Errorf("empty prefix")
	}
	start = max(0, min(start, len(prefix)-1))
	end = max(0, min(end, len(prefix)-1))
	result := prefix[end] - prefix[start]
	if mean != nil {
		result += *mean * float32(end-start)
	}
	return result, nil
}

// ContinuousLengthFeatures matches the boundary scorer: log1p(length), length
// divided by the document token length, and reciprocal sqrt(length).
func ContinuousLengthFeatures(start, end, tokenLength int) ([3]float32, error) {
	if tokenLength < 0 {
		return [3]float32{}, fmt.Errorf("negative document length")
	}
	length := float32(max(end-start, 1))
	return [3]float32{float32(math.Log1p(float64(length))), length / float32(max(tokenLength, 1)), 1 / float32(math.Sqrt(float64(length)))}, nil
}

// EndpointCompatibility uses the shared SIMD/Plan 9 dot dispatch. Model-level
// projection, scale and prior terms remain the caller's responsibility.
func EndpointCompatibility(start, end []float32) (float32, error) {
	if len(start) == 0 || len(start) != len(end) {
		return 0, fmt.Errorf("endpoint vector shape mismatch")
	}
	return simd.Sdot(start, end), nil
}

func Sigmoid(logit float32) float32 {
	if logit >= 0 {
		return 1 / (1 + float32(math.Exp(-float64(logit))))
	}
	e := float32(math.Exp(float64(logit)))
	return e / (1 + e)
}
