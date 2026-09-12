package audio

import (
	"context"
	"fmt"
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
	window     []float64
	cosine     []float64
	sine       []float64
	filters    []float64 // [bin, mel], matching Transformers (80 bands)
	filters128 []float64
}

// WhisperLogMel80 computes the Transformers WhisperFeatureExtractor contract:
// centered reflect-padded 400-point STFT, periodic Hann window, Slaney-normalized
// 80-bin mel projection, final-frame removal, log10 clamp and normalization.
// Input is expected to be an already right-padded 30-second 16 kHz chunk.
// For checked window/band validation use WhisperLogMel; invalid inputs here
// return nil, 0.
func WhisperLogMel80(samples []float32) ([]float32, int) {
	out, frames, _ := WhisperLogMel(samples, 80)
	return out, frames
}

// WhisperLogMel implements the 80- or 128-band WhisperFeatureExtractor contract.
// Input is mono 16 kHz PCM for one window (160..480000 samples). It does not
// resample, right-pad or truncate: callers own windowing. Output is mel-major,
// with floor(len(samples)/160) frames; sample-timeline mapping stays with callers.
// Non-finite inputs and invalid shapes are rejected before allocation/dispatch.
// This correctness-first DFT reuses the checked Plan 9 SIMD Ddot kernel; it is not
// the planned high-throughput FFT implementation.
func WhisperLogMel(samples []float32, numMels int) ([]float32, int, error) {
	return WhisperLogMelContext(context.Background(), samples, numMels)
}

// WhisperLogMelContext preserves WhisperLogMel's numerical operations, adding
// cancellation checks between frames and in bounded validation/normalisation
// blocks. One DFT frame, the bounded lookup-table sync.Once initialisation and
// reflect-padding copy are not interruptible. No partial features escape.
func WhisperLogMelContext(ctx context.Context, samples []float32, numMels int) ([]float32, int, error) {
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	if numMels != 80 && numMels != 128 {
		return nil, 0, fmt.Errorf("unsupported Whisper mel band count %d", numMels)
	}
	if len(samples) < whisperHop || len(samples) > 30*16000 {
		return nil, 0, fmt.Errorf("Whisper window requires 160..480000 mono samples")
	}
	for index, value := range samples {
		if index%16384 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, 0, err
			}
		}
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, 0, fmt.Errorf("non-finite Whisper waveform")
		}
	}
	allFrames := 1 + len(samples)/whisperHop
	frames := allFrames - 1 // WhisperFeatureExtractor deliberately drops this.
	if frames <= 0 {
		return nil, 0, nil
	}
	out := make([]float32, numMels*frames)
	nonzero := false
	for index, sample := range samples {
		if index%16384 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, 0, err
			}
		}
		if sample != 0 {
			nonzero = true
			break
		}
	}
	if !nonzero {
		for i := range out {
			if i%16384 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, 0, err
				}
			}
			out[i] = -1.5 // log10(1e-10), then (x+4)/4
		}
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		return out, frames, nil
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	whisperExactTables.Do(initWhisperExactTables)
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	filters := whisperExactTables.filters
	if numMels == 128 {
		filters = whisperExactTables.filters128
	}
	centered := reflectCenter(samples, whisperFFTSize/2)
	power := make([]float64, whisperBins)
	windowed := make([]float64, whisperFFTSize)
	maxLog := float32(-math.MaxFloat32)
	for frame := 0; frame < frames; frame++ {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
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
		for mel := 0; mel < numMels; mel++ {
			energy := float64(0)
			for bin := 0; bin < whisperBins; bin++ {
				energy += filters[bin*numMels+mel] * power[bin]
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
		if i%16384 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, 0, err
			}
		}
		if value < floor {
			value = floor
		}
		out[i] = (value + 4) / 4
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	return out, frames, nil
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
	t.filters128 = whisperSlaneyFiltersFor(128)
}

func whisperSlaneyFilters() []float64 { return whisperSlaneyFiltersFor(80) }

func whisperSlaneyFiltersFor(numMels int) []float64 {
	const sampleRate = 16000
	melMin := slaneyHzToMel(0)
	melMax := slaneyHzToMel(sampleRate / 2)
	centers := make([]float64, numMels+2)
	for i := range centers {
		mel := melMin + float64(i)*(melMax-melMin)/float64(numMels+1)
		centers[i] = slaneyMelToHz(mel)
	}
	filters := make([]float64, whisperBins*numMels)
	for bin := 0; bin < whisperBins; bin++ {
		frequency := float64(bin) * float64(sampleRate/2) / float64(whisperBins-1)
		for mel := 0; mel < numMels; mel++ {
			down := (frequency - centers[mel]) / (centers[mel+1] - centers[mel])
			up := (centers[mel+2] - frequency) / (centers[mel+2] - centers[mel+1])
			weight := math.Max(0, math.Min(down, up))
			weight *= 2 / (centers[mel+2] - centers[mel])
			filters[bin*numMels+mel] = weight
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
	// Repeated reflection matches numpy.pad for windows shorter than padding.
	period := 2 * (len(samples) - 1)
	reflectIndex := func(i int) int {
		i %= period
		if i < 0 {
			i += period
		}
		if i >= len(samples) {
			i = period - i
		}
		return i
	}
	for i := 0; i < padding; i++ {
		out[i] = samples[reflectIndex(i-padding)]
		out[padding+len(samples)+i] = samples[reflectIndex(len(samples)+i)]
	}
	return out
}
