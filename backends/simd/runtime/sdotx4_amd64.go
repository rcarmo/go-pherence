//go:build amd64

package simd

const HasSdotx4SIMD = true

//go:noescape
func sdotx4Asm(w, x []float32, stride int) (dot0 float32, dot1 float32, dot2 float32, dot3 float32)

func sdotx4(w, x []float32, stride int) (float32, float32, float32, float32) {
	// The AVX2 kernel is specialized for the hot DiffusionGemma expert path,
	// where hiddenSize=2816 and therefore len(w)%16==0. Keep odd tails on the
	// scalar fallback to avoid subtle tail-order differences in generic callers.
	if len(w)%16 != 0 || !HasDotAsm {
		return sdotx4Scalar(w, x, stride)
	}
	return sdotx4Asm(w, x, stride)
}
