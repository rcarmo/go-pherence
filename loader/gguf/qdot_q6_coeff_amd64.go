//go:build amd64

package gguf

import "golang.org/x/sys/cpu"

//go:noescape
func q6KCoeffDotAsm(q8 *[256]int8, coeff *[256]int16) int32

//go:noescape
func q6KExpandCoeffAsm(block *[210]byte, coeff *[256]int16)

//go:noescape
func q6KCoeffDot8Asm(q8 *[256]int8, coeff *[256]int16, out *[8]int32)

//go:noescape
func q6KBlockDotAsm(block *[210]byte, q8 *[256]int8) int32

//go:noescape
func q6KBlockDotVNNIAsm(block *[210]byte, q8 *[256]int8) int32

//go:noescape
func dotQ6KQ8KGemvVNNIAsm(raw []byte, y []q8KBlock, blocks int) float32

func q6KCoeffDot(q8 *[256]int8, coeff *[256]int16) int32 { return q6KCoeffDotAsm(q8, coeff) }

func q6KExpandCoeff(block *[210]byte, coeff *[256]int16) {
	q6KExpandCoeffAsm(block, coeff)
}

func q6KCoeffDot8(q8 *[256]int8, coeff *[256]int16, out *[8]int32) {
	q6KCoeffDot8Asm(q8, coeff, out)
}

func q6KBlockDot(block *[210]byte, q8 *[256]int8) int32 {
	if cpu.X86.HasAVXVNNI {
		return q6KBlockDotVNNIAsm(block, q8)
	}
	return q6KBlockDotAsm(block, q8)
}

func dotQ6KQ8KGemvVNNI(raw []byte, y []q8KBlock, blocks int) (float32, bool) {
	if !cpu.X86.HasAVXVNNI {
		return 0, false
	}
	return dotQ6KQ8KGemvVNNIAsm(raw, y, blocks), true
}

func q6KBlockDotAVX2(block *[210]byte, q8 *[256]int8) int32 {
	return q6KBlockDotAsm(block, q8)
}

func q6KBlockDotVNNI(block *[210]byte, q8 *[256]int8) (int32, bool) {
	if !cpu.X86.HasAVXVNNI {
		return 0, false
	}
	return q6KBlockDotVNNIAsm(block, q8), true
}
