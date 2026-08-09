//go:build amd64

package gguf

import (
	"github.com/rcarmo/go-pherence/half"
	"golang.org/x/sys/cpu"
)

//go:noescape
func quantizeQ8_0BlockAVX2(d *float32, qs *[qk8_0]int8, row *float32)

func quantizeQ8_0BlockSIMD(d *float32, qs *[qk8_0]int8, row []float32) bool {
	if len(row) != qk8_0 || !cpu.X86.HasAVX2 {
		return false
	}
	quantizeQ8_0BlockAVX2(d, qs, &row[0])
	*d = half.F16ToF32(half.F32ToF16(*d))
	return true
}
