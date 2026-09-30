//go:build amd64

package simd

import "golang.org/x/sys/cpu"

//go:noescape
func attentionValueRowAsm(dst, probabilities, values []float32, rows, width int)

func attentionValueRowTo(dst, probabilities, values []float32, rows, width int) {
	if cpu.X86.HasAVX2 && width%8 == 0 {
		attentionValueRowAsm(dst, probabilities, values, rows, width)
		return
	}
	attentionValueRowScalar(dst, probabilities, values, rows, width)
}
