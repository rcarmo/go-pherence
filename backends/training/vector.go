package training

import (
	"fmt"
	"math"
)

// VectorShape mode0=out=A+scale*B,mode1=slice A[offset:offset+n],
// mode2=pad source A into fresh n-element zero output at offset.
func VectorShape(n, source, offset int, mode uint32, scale float32) error {
	if n < 1 || n > 1<<26 || source < 1 || source > 1<<26 || offset < 0 || mode > 2 || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return fmt.Errorf("invalid training vector shape")
	}
	if (mode == 0 && (source != n || offset != 0)) || (mode == 1 && (offset > source || n > source-offset)) || (mode == 2 && (offset > n || source > n-offset)) {
		return fmt.Errorf("training vector range mismatch")
	}
	return nil
}
