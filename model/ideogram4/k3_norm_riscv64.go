//go:build riscv64

package ideogram4

import "math"

func k3RMSNormWeighted(dst []float32, x, weight []float32, eps float32) bool {
	if !k3Enabled() || len(dst) != len(x) || len(weight) != len(x) || len(x) == 0 {
		return false
	}
	// K3 runtime seam for weighted RMSNorm. The current body preserves exact
	// scalar semantics; replace with an RVV implementation using k3_isa.h macros
	// once the f32 reduction/vector multiply kernel is added.
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := float32(1 / math.Sqrt(ss/float64(len(x))+float64(eps)))
	for i := range x {
		dst[i] = x[i] * inv * weight[i]
	}
	return true
}
