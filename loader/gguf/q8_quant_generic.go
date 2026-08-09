//go:build !amd64

package gguf

func quantizeQ8_0BlockSIMD(_ *float32, _ *[qk8_0]int8, _ []float32) bool {
	return false
}
