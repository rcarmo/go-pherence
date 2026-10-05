package training

import "fmt"

// ReLUShape bounds an owned forward vector(mode0) or input VJP(mode1).
// Input derivative is zero at x<=0, matching the weighting MLP convention.
func ReLUShape(n int, mode uint32) error {
	if n < 1 || n > 1<<26 || mode > 1 {
		return fmt.Errorf("invalid training ReLU size/mode")
	}
	return nil
}
