//go:build !riscv64

package simd

func applyRoPEPartialAccel(x, freqs []float32, pos, numHeads, headDim, rotHalf int) bool {
	return false
}
