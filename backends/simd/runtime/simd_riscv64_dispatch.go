//go:build riscv64

package simd

//go:noescape
func sdotAsm(x, y []float32) float32

//go:noescape
func saxpyAsm(alpha float32, x []float32, y []float32)

func Sdot(x, y []float32) float32 {
	if len(x) > 0 && len(x) == len(y) && HasDotAsm {
		return sdotAsm(x, y)
	}
	return sdotScalar(x, y)
}

func Saxpy(alpha float32, x []float32, y []float32) {
	if len(x) == len(y) && HasDotAsm {
		saxpyAsm(alpha, x, y)
		return
	}
	saxpyScalar(alpha, x, y)
}
