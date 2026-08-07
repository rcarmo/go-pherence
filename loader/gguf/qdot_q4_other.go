//go:build !amd64

package gguf

func dotQ4_0Q8_0Packed(raw []byte, y []q8_0Block, blocks int) float32 {
	return dotQ4_0Q8_0Scalar(raw, y, blocks)
}
