//go:build !amd64 || !cgo

package llamaq4

import "fmt"

func DotQ4_0x8Q8_0x4VNNI(q4, q8 []byte, blocks int, out *[32]float32) error {
	return fmt.Errorf("llama Q4_0x8 AVX-VNNI experiment is unavailable")
}

func ProjectQ4_0x8Q8_0x4VNNI(q4, q8 []byte, rows, tokens, blocks int, out []float32) error {
	return fmt.Errorf("llama Q4_0x8 AVX-VNNI experiment is unavailable")
}
