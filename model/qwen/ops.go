package qwen

import "github.com/rcarmo/go-pherence/backends/simd/runtime"

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
	weightLen, ok := checkedMulInt(inDim, outDim)
	if inDim <= 0 || outDim <= 0 || !ok || len(out) < outDim || len(x) < inDim || len(w) < weightLen {
		return
	}
	simd.GemvRows(out[:outDim], x[:inDim], w[:weightLen], outDim, inDim)
}

func checkedMulInt(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if b != 0 && a > maxInt/b {
		return 0, false
	}
	return a * b, true
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
