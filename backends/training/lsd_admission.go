package training

import (
	"fmt"
	"math"
)

// LSDPredictionCutoff returns the largest nonnegative finite F32 difference
// magnitude accepted by the legacy F64 reduction/seed-overflow comparisons.
// The host computes this once; a device checks abs(F32(prediction-desired))
// against it. Binary search avoids double-rounded cutoff approximations.
func LSDPredictionCutoff(channels int, logvar float32) (float32, error) {
	if _, err := LSDLossShape(channels, 0); err != nil {
		return 0, err
	}
	if !residentFinite(logvar) || logvar < -80 || logvar > 80 {
		return 0, fmt.Errorf("invalid LSD admission logvar")
	}
	exp := math.Exp(float64(logvar))
	limit := math.Sqrt((float64(math.MaxFloat32) / 4) / float64(channels) / math.Max(1, exp))
	accepted := func(bits uint32) bool {
		d := float64(math.Float32frombits(bits))
		return d <= limit && 2*d*exp/float64(channels) <= float64(math.MaxFloat32)/4
	}
	low, high := uint32(0), uint32(0x7f7fffff)
	for low < high {
		mid := low + (high-low+1)/2
		if accepted(mid) {
			low = mid
		} else {
			high = mid - 1
		}
	}
	return math.Float32frombits(low), nil
}
