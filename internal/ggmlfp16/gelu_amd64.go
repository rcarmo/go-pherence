//go:build amd64

package ggmlfp16

import (
	"math"
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
	// The retained assembly narrows NaNs as infinity. Stop before the first
	// NaN-containing vector and let the scalar caller preserve quieted payloads.
	// Scan before writes so in-place gate/destination aliases remain correct.
	for i, v := range gate[:n] {
		if math.IsNaN(float64(v)) {
			n = i &^ 7
			break
		}
	}
	if n == 0 {
		return 0
	}
	_go_gelu_fp16_mul_avx2(unsafe.SliceData(dst), unsafe.SliceData(gate), unsafe.SliceData(up), n, (*byte)(unsafe.Pointer(&geluTable[0])))
	return n
}
