//go:build riscv64

package ideogram4

import (
	"math"

	simdruntime "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func k3RMSNormWeighted(dst []float32, x, weight []float32, eps float32) bool {
	if !k3Enabled() || len(dst) != len(x) || len(weight) != len(x) || len(x) == 0 {
		return false
	}
	// Compose existing RVV primitives: Snrm2 uses RVV dot where available,
	// VecScale and VecMul use RVV vector loops. This keeps the K3 path SIMD-backed
	// today while leaving room for a future fused row RMSNorm assembly kernel.
	ssqrt := simdruntime.Snrm2(x)
	inv := float32(1 / math.Sqrt(float64(ssqrt*ssqrt)/float64(len(x))+float64(eps)))
	simdruntime.VecScale(dst, x, inv)
	simdruntime.VecMul(dst, dst, weight)
	return true
}
