package audio

import "math"

const whisperFFT400Radix = 20

type whisperFFT400Plan struct {
	root20  [whisperFFT400Radix * whisperFFT400Radix]complex128
	twiddle [whisperFFTSize]complex128
}

// newWhisperFFT400Plan constructs the fixed 20x20 Cooley-Tukey plan used by the
// exact Whisper frontend. It has no mutable state and may be shared by callers.
func newWhisperFFT400Plan() whisperFFT400Plan {
	var p whisperFFT400Plan
	for k := 0; k < whisperFFT400Radix; k++ {
		for n := 0; n < whisperFFT400Radix; n++ {
			angle := -2 * math.Pi * float64(k*n) / whisperFFT400Radix
			p.root20[k*whisperFFT400Radix+n] = complex(math.Cos(angle), math.Sin(angle))
		}
	}
	for i := range p.twiddle {
		angle := -2 * math.Pi * float64(i) / whisperFFTSize
		p.twiddle[i] = complex(math.Cos(angle), math.Sin(angle))
	}
	return p
}

// powerSpectrum400 computes bins 0..200 of a real 400-point DFT. The 20x20
// factorisation performs 16,000 complex multiply-adds rather than 160,800 real
// dot-product terms. stage is caller-owned scratch and is overwritten. Results
// retain the frontend's complex64 rounding boundary before float64 magnitude.
func (p *whisperFFT400Plan) powerSpectrum400(power []float64, input []float64, stage []complex128) bool {
	if p == nil || len(power) != whisperBins || len(input) != whisperFFTSize || len(stage) != whisperFFTSize {
		return false
	}
	const radix = whisperFFT400Radix
	// n=n2+20*n1, k=k1+20*k2. First transform n1, then apply the
	// exp(-2pi*i*n2*k1/400) twiddle before transforming n2.
	for n2 := 0; n2 < radix; n2++ {
		for k1 := 0; k1 < radix; k1++ {
			var sum complex128
			roots := p.root20[k1*radix : (k1+1)*radix]
			for n1 := 0; n1 < radix; n1++ {
				sum += complex(input[n2+radix*n1], 0) * roots[n1]
			}
			stage[n2*radix+k1] = sum * p.twiddle[n2*k1]
		}
	}
	for k := 0; k < whisperBins; k++ {
		k1, k2 := k%radix, k/radix
		var sum complex128
		roots := p.root20[k2*radix : (k2+1)*radix]
		for n2 := 0; n2 < radix; n2++ {
			sum += stage[n2*radix+k1] * roots[n2]
		}
		r, i := float64(float32(real(sum))), float64(float32(imag(sum)))
		power[k] = r*r + i*i
	}
	return true
}
