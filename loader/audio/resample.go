package audio

import "math"

// Resample converts audio from srcRate to dstRate using linear interpolation.
// For production quality, a polyphase sinc filter should replace this.
func Resample(samples []float32, srcRate, dstRate int) []float32 {
	if srcRate == dstRate || len(samples) == 0 {
		return samples
	}
	ratio := float64(srcRate) / float64(dstRate)
	outLen := int(float64(len(samples)) / ratio)
	if outLen <= 0 {
		return nil
	}
	out := make([]float32, outLen)
	for i := range out {
		srcPos := float64(i) * ratio
		idx := int(srcPos)
		frac := float32(srcPos - float64(idx))
		if idx+1 < len(samples) {
			out[i] = samples[idx]*(1-frac) + samples[idx+1]*frac
		} else if idx < len(samples) {
			out[i] = samples[idx]
		}
	}
	return out
}

// ResampleSinc resamples using a windowed-sinc interpolation (quality=16 taps).
// This provides better frequency response than linear for speech.
func ResampleSinc(samples []float32, srcRate, dstRate int) []float32 {
	if srcRate == dstRate || len(samples) == 0 {
		return samples
	}

	const taps = 16
	ratio := float64(srcRate) / float64(dstRate)
	outLen := int(float64(len(samples)) / ratio)
	if outLen <= 0 {
		return nil
	}

	out := make([]float32, outLen)
	for i := range out {
		srcPos := float64(i) * ratio
		center := int(srcPos)

		var sum, wsum float64
		for j := center - taps; j <= center+taps; j++ {
			if j < 0 || j >= len(samples) {
				continue
			}
			x := srcPos - float64(j)
			// Lanczos-windowed sinc
			s := sinc(x) * lanczos(x, taps)
			sum += float64(samples[j]) * s
			wsum += s
		}
		if wsum != 0 {
			out[i] = float32(sum / wsum)
		}
	}
	return out
}

func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	px := math.Pi * x
	return math.Sin(px) / px
}

func lanczos(x float64, a int) float64 {
	if x == 0 {
		return 1
	}
	if x < float64(-a) || x > float64(a) {
		return 0
	}
	return sinc(x / float64(a))
}
