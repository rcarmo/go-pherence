// Package floatcmp provides small float-slice comparison helpers for tests.
package floatcmp

import "math"

// Close reports whether a and b have equal length and every element pair is
// within finite, nonnegative tol of each other. NaNs never compare close;
// infinities compare equal only when both have the same sign.
func Close(a, b []float32, tol float32) bool {
	if len(a) != len(b) || tol < 0 || math.IsNaN(float64(tol)) || math.IsInf(float64(tol), 0) {
		return false
	}
	for i := range a {
		if a[i] == b[i] {
			continue
		} // signed zeros and matching infinities
		// Compare widened operands; !(<=) fails closed for NaN rather than allowing
		// silent NaN equality or overflow in the float32 subtraction.
		if !(math.Abs(float64(a[i])-float64(b[i])) <= float64(tol)) {
			return false
		}
	}
	return true
}
