//go:build amd64

package gguf

import "golang.org/x/sys/cpu"

//go:noescape
func dotQ4_0Q8_0AVX2(raw []byte, y []q8_0Block, blocks int) float32

//go:noescape
func dotQ4_0Q8_0x4AVX2(raw []byte, rowBytes int, y []q8_0Block, blocks int, out *[4]float32)

//go:noescape
func dotQ4_0Q8_0x4VNNI(raw []byte, rowBytes int, y []q8_0Block, blocks int, out *[4]float32)

//go:noescape
func dotQ4_0Q8_0x8VNNI(raw []byte, rowBytes int, y []q8_0Block, corrections []q4Q8Correction, blocks int, out *[8]float32)

//go:noescape
func dotQ4_0Q8_0x4TokensAVX2(raw []byte, y []q8_0Block, blocks int, out *[4]float32)

//go:noescape
func dotQ4_0Q8_0Rows4Tokens2VNNI(raw []byte, rowBytes int, y []q8_0Block, blocks int, out *[8]float32)

//go:noescape
func dotQ4_0Q8_0Tokens8VNNI(raw []byte, y []q8_0Block, blocks, tokenStride, blockStride int, out *[8]float32)

func dotQ4_0Q8_0Packed(raw []byte, y []q8_0Block, blocks int) float32 {
	return dotQ4_0Q8_0AVX2(raw, y, blocks)
}

func dotQ4_0Q8_0Rows4(raw []byte, rowBytes int, y []q8_0Block, blocks int, out *[4]float32) {
	if cpu.X86.HasAVXVNNI {
		dotQ4_0Q8_0x4VNNI(raw, rowBytes, y, blocks, out)
		return
	}
	dotQ4_0Q8_0x4AVX2(raw, rowBytes, y, blocks, out)
}

func dotQ4_0Q8_0Rows4AVX2(raw []byte, rowBytes int, y []q8_0Block, blocks int, out *[4]float32) {
	dotQ4_0Q8_0x4AVX2(raw, rowBytes, y, blocks, out)
}

func supportsQ4_0Q8_0Rows8() bool {
	return cpu.X86.HasAVXVNNI
}

func dotQ4_0Q8_0Rows8VNNI(raw []byte, rowBytes int, y []q8_0Block, corrections []q4Q8Correction, blocks int, out *[8]float32) bool {
	if !cpu.X86.HasAVXVNNI {
		return false
	}
	dotQ4_0Q8_0x8VNNI(raw, rowBytes, y, corrections, blocks, out)
	return true
}

func dotQ4_0Q8_0Rows4VNNI(raw []byte, rowBytes int, y []q8_0Block, blocks int, out *[4]float32) bool {
	if !cpu.X86.HasAVXVNNI {
		return false
	}
	dotQ4_0Q8_0x4VNNI(raw, rowBytes, y, blocks, out)
	return true
}

func dotQ4_0Q8_0Tokens4(raw []byte, y []q8_0Block, blocks int, out *[4]float32) {
	dotQ4_0Q8_0x4TokensAVX2(raw, y, blocks, out)
}

func supportsQ4_0Q8_0Rows4Tokens2() bool {
	return cpu.X86.HasAVXVNNI
}

func dotQ4_0Q8_0Tokens8(raw []byte, y []q8_0Block, blocks int, out *[8]float32) bool {
	if !cpu.X86.HasAVXVNNI {
		return false
	}
	dotQ4_0Q8_0Tokens8VNNI(raw, y, blocks, blocks*36, 36, out)
	return true
}

func dotQ4_0Q8_0Tokens8Interleaved(raw []byte, y []q8_0Block, blocks int, out *[8]float32) bool {
	if !cpu.X86.HasAVXVNNI {
		return false
	}
	dotQ4_0Q8_0Tokens8VNNI(raw, y, blocks, 36, 8*36, out)
	return true
}

func dotQ4_0Q8_0Rows4Tokens2(raw []byte, rowBytes int, y []q8_0Block, blocks int, out *[8]float32) bool {
	if !supportsQ4_0Q8_0Rows4Tokens2() {
		return false
	}
	dotQ4_0Q8_0Rows4Tokens2VNNI(raw, rowBytes, y, blocks, out)
	return true
}
