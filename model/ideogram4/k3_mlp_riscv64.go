//go:build riscv64

package ideogram4

import simdruntime "github.com/rcarmo/go-pherence/backends/simd/runtime"

func k3SiLUTo(dst, x []float32) bool {
	if !k3Enabled() || len(dst) != len(x) || len(x) == 0 {
		return false
	}
	// K3 runtime seam for SiLU vector activation. Current body preserves scalar
	// semantics; replace with RVV approximation/table kernel when added.
	for i, v := range x {
		dst[i] = siluScalar(v)
	}
	return true
}

func k3MulTo(dst, a, b []float32) bool {
	if !k3Enabled() || len(dst) != len(a) || len(dst) != len(b) || len(dst) == 0 {
		return false
	}
	simdruntime.VecMul(dst, a, b)
	return true
}

func k3SiLUMulInPlace(gate, up []float32) bool {
	if !k3Enabled() || len(gate) != len(up) || len(gate) == 0 {
		return false
	}
	simdruntime.VecSiLUMul(gate, gate, up)
	return true
}
