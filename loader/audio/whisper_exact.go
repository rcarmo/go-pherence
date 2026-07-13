package audio

import (
	"math"
	"sync"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

const (
	whisperFFTSize = 400
	whisperHop     = 160
	whisperMels    = 80
	whisperBins    = whisperFFTSize/2 + 1
)

var whisperExactTables struct {
	sync.Once
	window  []float64
	cosine  []float64
	sine    []float64
	filters []float64 // [bin, mel], matching Transformers
}

// WhisperLogMel80 computes the Transformers WhisperFeatureExtractor contract:
// centered reflect-padded 400-point STFT, periodic Hann window, Slaney-normalized
// 80-bin mel projection, final-frame removal, log10 clamp and normalization.
// Input is expected to be an already right-padded 30-second 16 kHz chunk.
func WhisperLogMel80(samples []float32) ([]float32, int) {
	if len(samples) < 2 {
		return nil, 0
	}
	allFrames := 1 + len(samples)/whisperHop
	frames := allFrames - 1 // WhisperFeatureExtractor deliberately drops this.
	if frames <= 0 {
		return nil, 0
	}
	out := make([]float32, whisperMels*frames)
	nonzero := false
	for _, sample := range samples {
		if sample != 0 {
			nonzero = true
			break
		}
	}
	if !nonzero {
		for i := range out {
			out[i] = -1.5 // log10(1e-10), then (x+4)/4
		}
		return out, frames
	}
	whisperExactTables.Do(initWhisperExactTables)
	centered := reflectCenter(samples, whisperFFTSize/2)
	power := make([]float64, whisperBins)
	windowed := make([]float64, whisperFFTSize)
	maxLog := float32(-math.MaxFloat32)
	for frame := 0; frame < frames; frame++ {
		start := frame * whisperHop
		for sample := range windowed {
			windowed[sample] = float64(centered[start+sample]) * whisperExactTables.window[sample]
		}
		for bin := 0; bin < whisperBins; bin++ {
			basis := bin * whisperFFTSize
			real := simd.Ddot(windowed, whisperExactTables.cosine[basis:basis+whisperFFTSize])
			imag := -simd.Ddot(windowed, whisperExactTables.sine[basis:basis+whisperFFTSize])
			// Transformers stores each FFT result in complex64 before taking
			// its float64 magnitude, so retain that rounding boundary.
			r := float64(float32(real))
			i := float64(float32(imag))
			power[bin] = r*r + i*i
		}
		for mel := 0; mel < whisperMels; mel++ {
			energy := float64(0)
			for bin := 0; bin < whisperBins; bin++ {
				energy += whisperExactTables.filters[bin*whisperMels+mel] * power[bin]
			}
			if energy < 1e-10 {
				energy = 1e-10
			}
			value := float32(math.Log10(energy))
			out[mel*frames+frame] = value
			if value > maxLog {
				maxLog = value
			}
		}
	}
	floor := maxLog - 8
	for i, value := range out {
		if value < floor {
			value = floor
		}
		out[i] = (value + 4) / 4
	}
	return out, frames
}

func initWhisperExactTables() {
	t := &whisperExactTables
	t.window = make([]float64, whisperFFTSize)
	t.cosine = make([]float64, whisperBins*whisperFFTSize)
	t.sine = make([]float64, whisperBins*whisperFFTSize)
	for sample := 0; sample < whisperFFTSize; sample++ {
		t.window[sample] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(sample)/whisperFFTSize)
		for bin := 0; bin < whisperBins; bin++ {
			angle := 2 * math.Pi * float64(bin*sample) / whisperFFTSize
			t.cosine[bin*whisperFFTSize+sample] = math.Cos(angle)
			t.sine[bin*whisperFFTSize+sample] = math.Sin(angle)
		}
	}
	t.filters = whisperSlaneyFilters()
}

func whisperSlaneyFilters() []float64 {
	const sampleRate = 16000
	melMin := slaneyHzToMel(0)
	melMax := slaneyHzToMel(sampleRate / 2)
	centers := make([]float64, whisperMels+2)
	for i := range centers {
		mel := melMin + float64(i)*(melMax-melMin)/float64(whisperMels+1)
		centers[i] = slaneyMelToHz(mel)
	}
	filters := make([]float64, whisperBins*whisperMels)
	for bin := 0; bin < whisperBins; bin++ {
		frequency := float64(bin) * float64(sampleRate/2) / float64(whisperBins-1)
		for mel := 0; mel < whisperMels; mel++ {
			down := (frequency - centers[mel]) / (centers[mel+1] - centers[mel])
			up := (centers[mel+2] - frequency) / (centers[mel+2] - centers[mel+1])
			weight := math.Max(0, math.Min(down, up))
			weight *= 2 / (centers[mel+2] - centers[mel])
			filters[bin*whisperMels+mel] = weight
		}
	}
	return filters
}

func slaneyHzToMel(hz float64) float64 {
	if hz < 1000 {
		return 3 * hz / 200
	}
	return 15 + math.Log(hz/1000)*(27/math.Log(6.4))
}

func slaneyMelToHz(mel float64) float64 {
	if mel < 15 {
		return 200 * mel / 3
	}
	return 1000 * math.Exp((math.Log(6.4)/27)*(mel-15))
}

func reflectCenter(samples []float32, padding int) []float32 {
	out := make([]float32, len(samples)+2*padding)
	copy(out[padding:], samples)
	for i := 0; i < padding; i++ {
		out[padding-1-i] = samples[i+1]
		out[padding+len(samples)+i] = samples[len(samples)-2-i]
	}
	return out
}
