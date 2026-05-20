package kernels

// ApplyRoPE applies full-half rotary position embedding in-place.
func ApplyRoPE(x, freqs []float32, pos, numHeads, headDim int) {
	ApplyRoPEPartial(x, freqs, pos, numHeads, headDim, headDim/2)
}

// ApplyRoPEPartial applies RoPE with partial rotation.
// Only the first rotHalf pairs are rotated; remaining dims are untouched.
func ApplyRoPEPartial(x, freqs []float32, pos, numHeads, headDim, rotHalf int) {
	if pos < 0 || numHeads <= 0 || headDim <= 0 || rotHalf <= 0 || len(x) == 0 || len(freqs) == 0 {
		return
	}
	if rotHalf > headDim/2 {
		rotHalf = headDim / 2
	}
	maxHeads := len(x) / headDim
	if numHeads > maxHeads {
		numHeads = maxHeads
	}
	for h := 0; h < numHeads; h++ {
		for i := 0; i < rotHalf; i++ {
			freqOff := (pos*rotHalf + i) * 2
			if freqOff+1 >= len(freqs) {
				break
			}
			cos := freqs[freqOff]
			sin := freqs[freqOff+1]
			idx0 := h*headDim + i
			idx1 := h*headDim + i + rotHalf
			x0 := x[idx0]
			x1 := x[idx1]
			x[idx0] = x0*cos - x1*sin
			x[idx1] = x0*sin + x1*cos
		}
	}
}
