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

func dotQ4_0Q8_0Rows4AVX2(raw []byte, rowBytes int, y []q8_0Block, blocks int, out *[4]float32) {
	dotQ4_0Q8_0Rows4(raw, rowBytes, y, blocks, out)
}

func supportsQ4_0Q8_0Rows8() bool {
	return false
}

func dotQ4_0Q8_0Rows8VNNI(raw []byte, rowBytes int, y []q8_0Block, corrections []q4Q8Correction, blocks int, out *[8]float32) bool {
	return false
}

func dotQ4_0Q8_0Rows4VNNI(raw []byte, rowBytes int, y []q8_0Block, blocks int, out *[4]float32) bool {
	return false
}

func dotQ4_0Q8_0Tokens4(raw []byte, y []q8_0Block, blocks int, out *[4]float32) {
	for i := range out {
		out[i] = dotQ4_0Q8_0Scalar(raw, y[i*blocks:(i+1)*blocks], blocks)
	}
}

func supportsQ4_0Q8_0Rows4Tokens2() bool {
	return false
}

func dotQ4_0Q8_0Tokens8(raw []byte, y []q8_0Block, blocks int, out *[8]float32) bool {
	return false
}

func dotQ4_0Q8_0Tokens8SoA(raw []byte, y []q8_0Tile8, blocks int, out *[8]float32) bool {
	return false
}

func dotQ4_0Q8_0Tokens8Interleaved(raw []byte, y []q8_0Block, blocks int, out *[8]float32) bool {
	return false
}

func dotQ4_0Q8_0Rows4Tokens2(raw []byte, rowBytes int, y []q8_0Block, blocks int, out *[8]float32) bool {
	return false
}
