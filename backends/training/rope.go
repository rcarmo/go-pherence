package training

import (
	"fmt"
	"math"
)

// RoPEShape bounds adjacent-pair rotation of [rows,heads,dim]. The immutable
// table is [rows,dim/2,2] in cosine,sine order and shared across heads.
// mode0 applies R; mode1 applies R^T for input adjoints. Output is disjoint.
func RoPEShape(rows, heads, dim int, mode uint32) error {
	if rows < 1 || rows > 1024 || heads < 1 || heads > 32 || dim < 2 || dim > 256 || dim%2 != 0 || mode > 1 {
		return fmt.Errorf("invalid training RoPE shape/mode")
	}
	return nil
}

// RoPETableInto builds the pinned adjacent-pair transformer convention with
// host F64 trigonometry rounded once to F32. It never initialises a device.
// origin is the first absolute token position; no padded-key origin is inferred.
// Bounds and parameters are checked before writing. Caller must not mutate the
// resulting table while device work uses it. This is input preparation, not a
// fallback implementation of the device rotation.
func RoPETableInto(dst []float32, rows, dim, origin int, maxPeriod float64) error {
	if err := RoPEShape(rows, 1, dim, 0); err != nil {
		return err
	}
	if len(dst) != rows*dim || origin < 0 || origin > 1<<24-rows || maxPeriod <= 0 || math.IsNaN(maxPeriod) || math.IsInf(maxPeriod, 0) {
		return fmt.Errorf("invalid training RoPE table/position/period")
	}
	// Periods below one can overflow Pow at high frequencies; preflight every
	// frequency/angle before writing any caller storage.
	pairs := dim / 2
	for p := 0; p < pairs; p++ {
		f := math.Pow(maxPeriod, -float64(p)/float64(pairs))
		angle := float64(origin+rows-1) * f
		if math.IsNaN(angle) || math.IsInf(angle, 0) {
			return fmt.Errorf("training RoPE angle overflow")
		}
	}
	for row := 0; row < rows; row++ {
		for p := 0; p < pairs; p++ {
			angle := float64(origin+row) * math.Pow(maxPeriod, -float64(p)/float64(pairs))
			dst[row*dim+2*p] = float32(math.Cos(angle))
			dst[row*dim+2*p+1] = float32(math.Sin(angle))
		}
	}
	return nil
}
