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
func dotQ4_0Q8_0Rows4Tokens2LaneVNNI(q4Packed, q8Packed []byte, blocks int, lanes *[8][8]float32)

//go:noescape
func dotQ4_0Q8_0Tokens8VNNI(raw []byte, y []q8_0Block, blocks, tokenStride, blockStride int, out *[8]float32)

//go:noescape
func dotQ4_0Q8_0Tokens8SoAVNNI(raw []byte, y []q8_0Tile8, blocks int, out *[8]float32)

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

func dotQ4_0Q8_0Tokens8SoA(raw []byte, y []q8_0Tile8, blocks int, out *[8]float32) bool {
	if !cpu.X86.HasAVXVNNI {
		return false
	}
	dotQ4_0Q8_0Tokens8SoAVNNI(raw, y, blocks, out)
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

// dotQ4_0Q8_0Rows4Tokens2LaneTransposed is an experimental, non-production
// entry point. Both packed inputs contain 72 bytes per QK block. The assembly
// returns each output's eight legacy FP32 lane states after its final transpose.
func dotQ4_0Q8_0Rows4Tokens2LaneTransposed(q4Packed, q8Packed []byte, blocks int, lanes *[8][8]float32, out *[8]float32) bool {
	if out == nil || !dotQ4_0Q8_0Rows4Tokens2LaneStates(q4Packed, q8Packed, blocks, lanes) {
		return false
	}
	for i := range out {
		out[i] = reduceQ4_0Q8_0Lanes(&lanes[i])
	}
	return true
}

func dotQ4_0Q8_0Rows4Tokens2LaneStates(q4Packed, q8Packed []byte, blocks int, lanes *[8][8]float32) bool {
	if !cpu.X86.HasAVXVNNI || blocks < 0 || len(q4Packed) < blocks*72 || len(q8Packed) < blocks*72 || lanes == nil {
		return false
	}
	dotQ4_0Q8_0Rows4Tokens2LaneVNNI(q4Packed, q8Packed, blocks, lanes)
	return true
}

func reduceQ4_0Q8_0Lanes(lanes *[8]float32) float32 {
	r0 := lanes[0] + lanes[4]
	r1 := lanes[1] + lanes[5]
	r2 := lanes[2] + lanes[6]
	r3 := lanes[3] + lanes[7]
	r0 += r2
	r1 += r3
	return r0 + r1
}
