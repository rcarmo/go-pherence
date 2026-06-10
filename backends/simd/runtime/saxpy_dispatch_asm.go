//go:build amd64 || arm64 || riscv64

package simd

// Saxpy computes y[i] += alpha*x[i] using assembly when available.
func Saxpy(alpha float32, x []float32, y []float32) {
	if len(x) == len(y) && HasDotAsm {
		saxpyAsm(alpha, x, y)
		return
	}
	saxpyScalar(alpha, x, y)
}
