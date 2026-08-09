//go:build amd64

package simd

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

//go:noescape
func _go_rope_partial_avx2(x, freq *float32, heads, headDim, rotHalf int)

func applyRoPEPartialAccel(x, freqs []float32, pos, numHeads, headDim, rotHalf int) bool {
	if !cpu.X86.HasAVX2 || pos < 0 || numHeads <= 0 || headDim <= 0 || rotHalf <= 0 || rotHalf > headDim/2 {
		return false
	}
	if numHeads > len(x)/headDim || pos > (len(freqs)/2)/rotHalf-1 {
		return false
	}
	freqBase := pos * rotHalf * 2
	_go_rope_partial_avx2(unsafe.SliceData(x), unsafe.SliceData(freqs[freqBase:]), numHeads, headDim, rotHalf)
	return true
}
