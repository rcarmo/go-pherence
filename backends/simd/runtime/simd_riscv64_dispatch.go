//go:build riscv64

package simd

//go:noescape
func sdotAsm(x, y []float32) float32

//go:noescape
func saxpyAsm(alpha float32, x []float32, y []float32)

func Sdot(x, y []float32) float32 {
	if len(x) > 0 && len(x) == len(y) && HasDotAsm {
		// m4 (4-register group) is ~1.4x the m1 kernel on the K1's
		// overhead/latency-bound in-order vector pipe.
		return sdotM4Asm(x, y)
	}
	return sdotScalar(x, y)
}
