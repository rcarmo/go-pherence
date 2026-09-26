//go:build amd64 || arm64

package simd

//go:noescape
func vecMulAddAsm(dst, a, b []float32)

func vecMulAdd(dst, a, b []float32) {
	if HasVecAsm {
		// A multiple of eight is valid for both AVX2 and NEON kernels.
		n := len(dst) &^ 7
		if n != 0 {
			vecMulAddAsm(dst[:n], a[:n], b[:n])
		}
		vecMulAddScalar(dst[n:], a[n:], b[n:])
		return
	}
	vecMulAddScalar(dst, a, b)
}
