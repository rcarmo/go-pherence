//go:build !amd64

package simd

func siluFinish(dst, gate, up, exp []float32) {
	for i := range dst {
		dst[i] = (gate[i] / (1 + exp[i])) * up[i]
	}
}
