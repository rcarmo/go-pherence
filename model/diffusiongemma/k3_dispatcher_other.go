//go:build !riscv64

package diffusiongemma

func k3Enabled() bool { return false }

func k3Dot(a, b []float32) float32 { return dot(a, b) }

func k3SoftmaxInPlace(x []float32) { softmaxInPlace(x) }

func k3SiLU(x float32) float32 { return siluScalar(x) }

func k3SaxpyV(w float32, v, out []float32) {
	for i := range v {
		out[i] += w * v[i]
	}
}
