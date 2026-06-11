//go:build riscv64

package ideogram4

import "math"

func k3LayerNormNoAffine(dst, x []float32, eps float32) bool {
	if !k3Enabled() || len(dst) != len(x) || len(x) == 0 {
		return false
	}
	// K3 runtime seam for final non-affine LayerNorm. Current body preserves
	// scalar semantics; replace with RVV row LayerNorm assembly.
	n := len(x)
	var mean float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= float64(n)
	var variance float64
	for _, v := range x {
		d := float64(v) - mean
		variance += d * d
	}
	variance /= float64(n)
	inv := float32(1 / math.Sqrt(variance+float64(eps)))
	mf := float32(mean)
	for i, v := range x {
		dst[i] = (v - mf) * inv
	}
	return true
}
