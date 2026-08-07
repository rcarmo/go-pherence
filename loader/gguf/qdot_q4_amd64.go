//go:build amd64

package gguf

//go:noescape
func dotQ4_0Q8_0AVX2(raw []byte, y []q8_0Block, blocks int) float32

//go:noescape
func dotQ4_0Q8_0x4AVX2(raw []byte, rowBytes int, y []q8_0Block, blocks int, out *[4]float32)

func dotQ4_0Q8_0Packed(raw []byte, y []q8_0Block, blocks int) float32 {
	return dotQ4_0Q8_0AVX2(raw, y, blocks)
}

func dotQ4_0Q8_0Rows4(raw []byte, rowBytes int, y []q8_0Block, blocks int, out *[4]float32) {
	dotQ4_0Q8_0x4AVX2(raw, rowBytes, y, blocks, out)
}
