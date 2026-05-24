package whisper

// linearForwardOpt computes out = x @ W^T + bias with loop reordering for cache friendliness.
// x: [seqLen, inDim], W: [outDim, inDim], bias: [outDim]
// Returns [seqLen, outDim].
func linearForwardOpt(x, weight, bias []float32, seqLen, inDim, outDim int) []float32 {
	out := make([]float32, seqLen*outDim)

	// For single-token (seqLen=1), use simple GEMV
	if seqLen == 1 {
		for o := 0; o < outDim; o++ {
			var sum float32
			wOff := o * inDim
			// Unroll by 4
			d := 0
			for ; d+3 < inDim; d += 4 {
				sum += x[d]*weight[wOff+d] +
					x[d+1]*weight[wOff+d+1] +
					x[d+2]*weight[wOff+d+2] +
					x[d+3]*weight[wOff+d+3]
			}
			for ; d < inDim; d++ {
				sum += x[d] * weight[wOff+d]
			}
			if bias != nil && o < len(bias) {
				sum += bias[o]
			}
			out[o] = sum
		}
		return out
	}

	// For batched (seqLen > 1), use tiled approach
	const tile = 4
	for t := 0; t < seqLen; t++ {
		xOff := t * inDim
		oOff := t * outDim
		o := 0
		for ; o+tile-1 < outDim; o += tile {
			var s0, s1, s2, s3 float32
			w0 := (o + 0) * inDim
			w1 := (o + 1) * inDim
			w2 := (o + 2) * inDim
			w3 := (o + 3) * inDim
			for d := 0; d < inDim; d++ {
				xd := x[xOff+d]
				s0 += xd * weight[w0+d]
				s1 += xd * weight[w1+d]
				s2 += xd * weight[w2+d]
				s3 += xd * weight[w3+d]
			}
			if bias != nil {
				s0 += bias[o+0]
				s1 += bias[o+1]
				s2 += bias[o+2]
				s3 += bias[o+3]
			}
			out[oOff+o+0] = s0
			out[oOff+o+1] = s1
			out[oOff+o+2] = s2
			out[oOff+o+3] = s3
		}
		for ; o < outDim; o++ {
			var sum float32
			wOff := o * inDim
			for d := 0; d < inDim; d++ {
				sum += x[xOff+d] * weight[wOff+d]
			}
			if bias != nil && o < len(bias) {
				sum += bias[o]
			}
			out[oOff+o] = sum
		}
	}
	return out
}
