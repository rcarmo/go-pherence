package tensor

import "math"

// AttentiveStatPool computes attentive statistics pooling over the time dimension.
// Input: h [channels, length] channel-first flat.
// Returns [channels*2] (weighted mean ++ weighted std).
//
// The attention mechanism computes per-timestep weights:
//
//	attn_hidden = tanh(W @ h_t + b)      W: [attnDim, channels], b: [attnDim]
//	score_t = V @ attn_hidden + vBias     V: [attnDim], vBias: scalar
//	weights = softmax(scores)
//
// Then weighted statistics:
//
//	mean_c = sum_t(w_t * h[c,t])
//	std_c  = sqrt(sum_t(w_t * h[c,t]^2) - mean_c^2)
func AttentiveStatPool(h []float32, channels, length int, W, b, V []float32, vBias float32, attnDim int) []float32 {
	if channels <= 0 || length <= 0 {
		return nil
	}

	// Compute attention scores per timestep
	scores := make([]float32, length)
	for t := 0; t < length; t++ {
		var score float32
		for a := 0; a < attnDim; a++ {
			// W @ h_t + b
			var sum float32
			for c := 0; c < channels; c++ {
				if a*channels+c < len(W) {
					sum += h[c*length+t] * W[a*channels+c]
				}
			}
			if b != nil && a < len(b) {
				sum += b[a]
			}
			// tanh
			hidden := float32(math.Tanh(float64(sum)))
			// V[a] * hidden
			if a < len(V) {
				score += V[a] * hidden
			}
		}
		scores[t] = score + vBias
	}

	// Softmax
	softmaxF32(scores)

	// Weighted mean and std
	out := make([]float32, channels*2)
	for c := 0; c < channels; c++ {
		var wMean, wSq float64
		for t := 0; t < length; t++ {
			v := float64(h[c*length+t])
			w := float64(scores[t])
			wMean += v * w
			wSq += v * v * w
		}
		out[c] = float32(wMean)
		variance := wSq - wMean*wMean
		if variance < 0 {
			variance = 0
		}
		out[channels+c] = float32(math.Sqrt(variance))
	}

	return out
}

// MeanStdPool computes simple mean and standard deviation pooling (no attention).
// Input: h [channels, length] channel-first flat.
// Returns [channels*2] (mean ++ std).
func MeanStdPool(h []float32, channels, length int) []float32 {
	if channels <= 0 || length <= 0 {
		return nil
	}
	out := make([]float32, channels*2)
	for c := 0; c < channels; c++ {
		var sum, sq float64
		for t := 0; t < length; t++ {
			v := float64(h[c*length+t])
			sum += v
			sq += v * v
		}
		mean := sum / float64(length)
		variance := sq/float64(length) - mean*mean
		if variance < 0 {
			variance = 0
		}
		out[c] = float32(mean)
		out[channels+c] = float32(math.Sqrt(variance))
	}
	return out
}

func softmaxF32(x []float32) {
	if len(x) == 0 {
		return
	}
	max := x[0]
	for _, v := range x[1:] {
		if v > max {
			max = v
		}
	}
	var sum float32
	for i, v := range x {
		e := float32(math.Exp(float64(v - max)))
		x[i] = e
		sum += e
	}
	if sum > 0 {
		for i := range x {
			x[i] /= sum
		}
	}
}
