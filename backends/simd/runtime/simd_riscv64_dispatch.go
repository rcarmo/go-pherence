//go:build riscv64

package simd

//go:noescape
func sdotAsm(x, y []float32) float32

func Sdot(x, y []float32) float32 {
	if len(x) > 0 && len(x) == len(y) && HasDotAsm {
		return sdotAsm(x, y)
	}
	return sdotScalar(x, y)
}
