package speaker

import "math"

// Embed computes a 192-dimensional speaker embedding for normalized 80-bin
// log-mel features shaped [80, frames] in channel-first layout. This implements
// the SpeechBrain ECAPA-TDNN topology in inference mode. It intentionally does
// not perform Fbank extraction or sentence mean normalization; callers should
// provide features matching the downloaded checkpoint's preprocessing.
func (m *SpeechBrainECAPA) Embed(mel []float32, frames int) []float32 {
	if m == nil || frames <= 0 || len(mel) < 80*frames {
		return nil
	}
	x := tdnnForward(m.Conv0, mel, 80, frames, 1)
	blockInputs := x
	blockOuts := make([][]float32, 0, len(m.Blocks))
	for i := range m.Blocks {
		b := &m.Blocks[i]
		y := tdnnForward(b.TDNN1, blockInputs, 1024, frames, 1)
		chunks := splitChannels(y, 1024, frames, 8)
		var prev []float32
		merged := make([][]float32, 0, 8)
		for j, chunk := range chunks {
			if j == 0 {
				prev = chunk
			} else if j == 1 {
				prev = tdnnForward(b.Res2Net[j-1], chunk, 128, frames, 2+i)
			} else {
				prev = tdnnForward(b.Res2Net[j-1], addSame(chunk, prev), 128, frames, 2+i)
			}
			merged = append(merged, prev)
		}
		y = concatChannels(merged, frames)
		y = tdnnForward(b.TDNN2, y, 1024, frames, 1)
		y = seForward(b.SE, y, 1024, frames)
		blockInputs = addSame(blockInputs, y)
		blockOuts = append(blockOuts, blockInputs)
	}
	multi := concatChannels(blockOuts, frames)
	x = tdnnForward(m.MFA, multi, 3072, frames, 1)
	pooled := aspForward(m.ASP, x, 3072, frames)
	pooled = batchNormForward(m.ASPBN, pooled, 6144, 1)
	out := conv1dForward(m.FC, pooled, 6144, 1, 1)
	return l2Normalize(out)
}

func tdnnForward(layer TDNNLayer, input []float32, inCh, frames, dilation int) []float32 {
	y := conv1dForwardDilated(layer.Conv, input, inCh, frames, dilation)
	relu(y)
	y = batchNormForward(layer.Norm, y, len(layer.Norm.Weight), frames)
	return y
}

func conv1dForward(c Conv1D, input []float32, inCh, frames, dilation int) []float32 {
	return conv1dForwardDilated(c, input, inCh, frames, dilation)
}

func conv1dForwardDilated(c Conv1D, input []float32, inCh, frames, dilation int) []float32 {
	if len(c.Shape) != 3 || frames <= 0 || inCh <= 0 {
		return nil
	}
	outCh, kernel := c.Shape[0], c.Shape[2]
	pad := dilation * (kernel / 2)
	out := make([]float32, outCh*frames)
	for oc := 0; oc < outCh; oc++ {
		for t := 0; t < frames; t++ {
			sum := float32(0)
			if oc < len(c.Bias) {
				sum = c.Bias[oc]
			}
			for ic := 0; ic < inCh; ic++ {
				wBase := (oc*inCh + ic) * kernel
				for k := 0; k < kernel; k++ {
					it := t + k*dilation - pad
					if it < 0 || it >= frames || wBase+k >= len(c.Weight) || ic*frames+it >= len(input) {
						continue
					}
					sum += input[ic*frames+it] * c.Weight[wBase+k]
				}
			}
			out[oc*frames+t] = sum
		}
	}
	return out
}

func batchNormForward(b BatchNorm1D, input []float32, channels, frames int) []float32 {
	out := make([]float32, len(input))
	const eps = 1e-5
	for c := 0; c < channels; c++ {
		weight, bias := float32(1), float32(0)
		mean, variance := float32(0), float32(1)
		if c < len(b.Weight) {
			weight = b.Weight[c]
		}
		if c < len(b.Bias) {
			bias = b.Bias[c]
		}
		if c < len(b.RunningMean) {
			mean = b.RunningMean[c]
		}
		if c < len(b.RunningVar) {
			variance = b.RunningVar[c]
		}
		scale := weight / float32(math.Sqrt(float64(variance+eps)))
		for t := 0; t < frames; t++ {
			idx := c*frames + t
			if idx < len(input) {
				out[idx] = (input[idx]-mean)*scale + bias
			}
		}
	}
	return out
}

func seForward(se SEBlock1D, input []float32, channels, frames int) []float32 {
	avg := make([]float32, channels)
	for c := 0; c < channels; c++ {
		var sum float32
		for t := 0; t < frames; t++ {
			sum += input[c*frames+t]
		}
		avg[c] = sum / float32(frames)
	}
	down := conv1dForward(se.Conv1, avg, channels, 1, 1)
	relu(down)
	up := conv1dForward(se.Conv2, down, 128, 1, 1)
	out := make([]float32, len(input))
	for c := 0; c < channels; c++ {
		scale := float32(0.5)
		if c < len(up) {
			scale = sigmoid(up[c])
		}
		for t := 0; t < frames; t++ {
			out[c*frames+t] = input[c*frames+t] * scale
		}
	}
	return out
}

func aspForward(asp SpeechBrainASP, input []float32, channels, frames int) []float32 {
	mean, std := channelMeanStd(input, channels, frames)
	global := make([]float32, channels*3*frames)
	for c := 0; c < channels; c++ {
		copy(global[c*frames:(c+1)*frames], input[c*frames:(c+1)*frames])
		for t := 0; t < frames; t++ {
			global[(channels+c)*frames+t] = mean[c]
			global[(2*channels+c)*frames+t] = std[c]
		}
	}
	att := tdnnForward(asp.TDNN, global, channels*3, frames, 1)
	for i := range att {
		att[i] = float32(math.Tanh(float64(att[i])))
	}
	att = conv1dForward(asp.Conv, att, 128, frames, 1)
	softmaxTimePerChannel(att, channels, frames)
	pooledMean := make([]float32, channels)
	pooledStd := make([]float32, channels)
	for c := 0; c < channels; c++ {
		var m, sq float64
		for t := 0; t < frames; t++ {
			w := float64(att[c*frames+t])
			v := float64(input[c*frames+t])
			m += w * v
			sq += w * v * v
		}
		pooledMean[c] = float32(m)
		variance := sq - m*m
		if variance < 1e-12 {
			variance = 1e-12
		}
		pooledStd[c] = float32(math.Sqrt(variance))
	}
	return append(pooledMean, pooledStd...)
}

func channelMeanStd(input []float32, channels, frames int) ([]float32, []float32) {
	mean := make([]float32, channels)
	std := make([]float32, channels)
	for c := 0; c < channels; c++ {
		var sum, sq float64
		for t := 0; t < frames; t++ {
			v := float64(input[c*frames+t])
			sum += v
			sq += v * v
		}
		m := sum / float64(frames)
		mean[c] = float32(m)
		variance := sq/float64(frames) - m*m
		if variance < 1e-12 {
			variance = 1e-12
		}
		std[c] = float32(math.Sqrt(variance))
	}
	return mean, std
}

func splitChannels(input []float32, channels, frames, parts int) [][]float32 {
	chunk := channels / parts
	out := make([][]float32, parts)
	for p := 0; p < parts; p++ {
		out[p] = make([]float32, chunk*frames)
		copy(out[p], input[p*chunk*frames:(p+1)*chunk*frames])
	}
	return out
}

func concatChannels(parts [][]float32, frames int) []float32 {
	total := 0
	for _, p := range parts {
		total += len(p)
	}
	out := make([]float32, 0, total)
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

func addSame(a, b []float32) []float32 {
	out := make([]float32, len(a))
	copy(out, a)
	for i := range out {
		if i < len(b) {
			out[i] += b[i]
		}
	}
	return out
}

func softmaxTimePerChannel(x []float32, channels, frames int) {
	for c := 0; c < channels; c++ {
		base := c * frames
		maxv := float32(math.Inf(-1))
		for t := 0; t < frames; t++ {
			if x[base+t] > maxv {
				maxv = x[base+t]
			}
		}
		var sum float64
		for t := 0; t < frames; t++ {
			v := math.Exp(float64(x[base+t] - maxv))
			x[base+t] = float32(v)
			sum += v
		}
		if sum == 0 {
			continue
		}
		for t := 0; t < frames; t++ {
			x[base+t] /= float32(sum)
		}
	}
}

func l2Normalize(x []float32) []float32 {
	var norm float64
	for _, v := range x {
		norm += float64(v) * float64(v)
	}
	if norm == 0 {
		return x
	}
	scale := float32(1 / math.Sqrt(norm))
	for i := range x {
		x[i] *= scale
	}
	return x
}
