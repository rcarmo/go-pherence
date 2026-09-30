//go:build !amd64

package simd

func attentionValueRowTo(dst, probabilities, values []float32, rows, width int) {
	attentionValueRowScalar(dst, probabilities, values, rows, width)
}
