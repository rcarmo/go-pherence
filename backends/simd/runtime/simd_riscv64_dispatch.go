//go:build riscv64

package simd

func Sdot(x, y []float32) float32 {
	if len(x) > 0 && len(x) == len(y) && HasDotAsm {
		// m4 (4-register group) is ~1.4x the m1 kernel on the K1's
		// overhead/latency-bound in-order vector pipe.
		return sdotM4Asm(x, y)
	}
	return sdotScalar(x, y)
}
