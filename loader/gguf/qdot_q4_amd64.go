//go:build amd64

package gguf

//go:noescape
func dotQ4_0Q8_0AVX2(raw []byte, y []q8_0Block, blocks int) float32

func dotQ4_0Q8_0Packed(raw []byte, y []q8_0Block, blocks int) float32 {
	return dotQ4_0Q8_0AVX2(raw, y, blocks)
}
