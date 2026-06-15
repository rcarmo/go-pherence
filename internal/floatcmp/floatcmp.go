// Package floatcmp provides small float-slice comparison helpers for tests.
package floatcmp

import "math"

// Close reports whether a and b have equal length and every element pair is
// within tol of each other.
func Close(a, b []float32, tol float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if float32(math.Abs(float64(a[i]-b[i]))) > tol {
			return false
		}
	}
	return true
}
