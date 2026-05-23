//go:build riscv64

package simd

//go:noescape
func ropePartialHeadAsm(x0, x1, freqs []float32, n int)

func applyRoPEPartialAccel(x, freqs []float32, pos, numHeads, headDim, rotHalf int) bool {
	if !RuntimeCapabilities().HasRVV || pos < 0 || numHeads <= 0 || headDim <= 0 || rotHalf <= 0 || rotHalf > headDim/2 {
		return false
	}
	maxHeads := len(x) / headDim
	if numHeads > maxHeads {
		numHeads = maxHeads
	}
	freqOff := pos * rotHalf * 2
	if numHeads <= 0 || freqOff < 0 || freqOff+rotHalf*2 > len(freqs) {
		return false
	}
	freqRow := freqs[freqOff : freqOff+rotHalf*2]
	for h := 0; h < numHeads; h++ {
		base := h * headDim
		x0 := x[base : base+rotHalf]
		x1 := x[base+rotHalf : base+2*rotHalf]
		ropePartialHeadAsm(x0, x1, freqRow, rotHalf)
	}
	return true
}
