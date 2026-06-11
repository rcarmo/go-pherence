//go:build riscv64

package ideogram4

import (
	"math"

	simdruntime "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func k3LayerNormNoAffine(dst, x []float32, eps float32) bool {
	if !k3Enabled() || len(dst) != len(x) || len(x) == 0 {
		return false
	}
	// K3 runtime seam for final non-affine LayerNorm. Centering still uses a
	// scalar mean/subtract loop, but the norm and final scale use existing RVV
	// primitives. Replace with a fused RVV row LayerNorm assembly kernel later.
	n := len(x)
	var mean float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= float64(n)
	mf := float32(mean)
	for i, v := range x {
		dst[i] = v - mf
	}
	ssqrt := simdruntime.Snrm2(dst)
	inv := float32(1 / math.Sqrt(float64(ssqrt*ssqrt)/float64(n)+float64(eps)))
	simdruntime.VecScale(dst, dst, inv)
	return true
}
