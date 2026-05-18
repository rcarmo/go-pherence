package qwen

import "github.com/rcarmo/go-pherence/backends/simd"

func rmsNormInPlace(x, weight []float32, eps float32) {
	simd.RMSNorm(x, weight, eps)
}

func simdDot(a, b []float32) float32 {
	if len(a) >= 8 {
		return simd.Sdot(a, b)
	}
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func gemvNT(out, x []float32, w []float32, inDim, outDim int) {
	for i := range out {
		out[i] = 0
	}
	if inDim <= 0 || outDim <= 0 || len(out) < outDim || len(x) < inDim || len(w) < inDim*outDim {
		return
	}
	for j := 0; j < outDim; j++ {
		sum := float32(0)
		row := w[j*inDim : (j+1)*inDim]
		if inDim >= 8 {
			sum = simd.Sdot(x, row)
		} else {
			for p := 0; p < inDim; p++ {
				sum += x[p] * row[p]
			}
		}
		out[j] = sum
	}
}

func applyRoPEPartial(x, freqs []float32, pos, numHeads, headDim, rotHalf int) {
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
