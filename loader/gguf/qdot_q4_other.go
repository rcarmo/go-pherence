//go:build !amd64

package gguf

func dotQ4_0Q8_0Packed(raw []byte, y []q8_0Block, blocks int) float32 {
	return dotQ4_0Q8_0Scalar(raw, y, blocks)
}

func dotQ4_0Q8_0Rows4(raw []byte, rowBytes int, y []q8_0Block, blocks int, out *[4]float32) {
	for i := range out {
		out[i] = dotQ4_0Q8_0Scalar(raw[i*rowBytes:(i+1)*rowBytes], y, blocks)
	}
}
