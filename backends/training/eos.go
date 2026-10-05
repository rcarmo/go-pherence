package training

import (
	"fmt"
	"math"
)

// EOSShape admits stable BCE per-position losses (mode0), weighted/mean logit
// gradients (mode1), or a compensated mean reduction of losses (mode2).
// Labels are -1 for ignored, 0/1 for binary target. Input/output disjoint.
func EOSShape(n, active int, mode uint32, weight float32) (output int, err error) {
	if n < 1 || n > 1024 || active < 1 || active > n || mode > 2 || weight < 0 || math.IsNaN(float64(weight)) || math.IsInf(float64(weight), 0) {
		return 0, fmt.Errorf("invalid training EOS shape/count/mode/weight")
	}
	output = n
	if mode == 2 {
		output = 1
	}
	return output, nil
}
