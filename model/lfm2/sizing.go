package lfm2

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// Layout fields use nonnegative int counts. -1 marks an invalid calculation;
// constructors call Validate before returning and validators reject sentinels.
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
func sizeBytes(values ...int) (int64, error) {
	n := int64(1)
	for _, v := range values {
		if v < 0 || v != 0 && n > math.MaxInt64/int64(v) {
			return 0, fmt.Errorf("invalid/overflowing LFM2 byte size")
		}
		n *= int64(v)
	}
	return n, nil
}
