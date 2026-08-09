//go:build amd64

package ggmlfp16

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

//go:noescape
func _go_gelu_fp16_mul_avx2(dst, gate, up *float32, n int, table *byte)

func cpuHasF16C() bool

var hasF16C = cpuHasF16C()

func geluFP16MulSIMD(dst, gate, up []float32) int {
	n := len(dst) &^ 7
	if n == 0 || !cpu.X86.HasAVX2 || !hasF16C {
		return 0
	}
	_go_gelu_fp16_mul_avx2(unsafe.SliceData(dst), unsafe.SliceData(gate), unsafe.SliceData(up), n, (*byte)(unsafe.Pointer(&geluTable[0])))
	return n
}
