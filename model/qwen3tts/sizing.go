package qwen3tts

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// Counts are nonnegative. Preserve an invalid sentinel during construction;
// validators must reject it before comparing computed and declared sizes.
func sizeProduct(values ...int) int {
	n := 1
	for _, v := range values {
		var ok bool
		n, ok = checked.MulInt(n, v)
		if !ok {
			return -1
		}
	}
	return n
}
func sizeSum(values ...int) int {
	n := 0
	for _, v := range values {
		var ok bool
		n, ok = checked.AddInt(n, v)
		if !ok {
			return -1
		}
	}
	return n
}
func nonnegativeSizes(values ...int) bool {
	for _, v := range values {
		if v < 0 {
			return false
		}
	}
	return true
}
func sizeCount(values ...int) (int, error) {
	n := sizeProduct(values...)
	if n < 0 {
		return 0, fmt.Errorf("Qwen3-TTS size overflow or negative factor: %v", values)
	}
	return n, nil
}
func sizeBytes(values ...int) (int64, error) {
	n := int64(1)
	for _, v := range values {
		if v < 0 || v > 0 && n > math.MaxInt64/int64(v) {
			return 0, fmt.Errorf("Qwen3-TTS byte size overflow or negative factor: %v", values)
		}
		n *= int64(v)
	}
	return n, nil
}
func finitePositive(v float64) bool { return v > 0 && !math.IsInf(v, 0) && !math.IsNaN(v) }
