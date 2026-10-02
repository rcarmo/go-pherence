package audio

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"sync"
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
	cosine     []float64 // [bin, sample], retained direct-DFT oracle
	sine       []float64
	fft400     whisperFFT400Plan
	filters    []float64 // [bin, mel], matching Transformers (80 bands)
	filters128 []float64
	// Per-mel [lo, hi) bin spans outside which the filter is exactly zero.
	spans, spans128 [][2]int
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
// A fixed mixed-radix FFT400 computes the spectrum; direct DFT tables remain a
// test oracle. The float32-complex rounding boundary remains explicit.
func WhisperLogMel(samples []float32, numMels int) ([]float32, int, error) {
	return WhisperLogMelContext(context.Background(), samples, numMels)
}

// WhisperLogMelContext preserves WhisperLogMel's numerical operations, adding
// cancellation checks between frames and in bounded validation/normalisation
// blocks. One DFT frame, the bounded lookup-table sync.Once initialisation and
// reflect-padding copy are not interruptible. No partial features escape.
func WhisperLogMelContext(ctx context.Context, samples []float32, numMels int) ([]float32, int, error) {
	out, frames, _, err := whisperLogMel(ctx, samples, numMels, nil, false)
	return out, frames, err
}

// WhisperLogMelClipFloorContext is WhisperLogMelContext with whisper.cpp's
// whole-clip clamp: floor = max(window max, clipMaxLog) - 8, where clipMaxLog
// is the raw log10 maximum over every window of the clip (see
// WhisperLogMelMaxContext). Values above the floor are unchanged. Explicit
// compatibility only; frames at window edges still use reflect padding.
func WhisperLogMelClipFloorContext(ctx context.Context, samples []float32, numMels int, clipMaxLog float32) ([]float32, int, error) {
	if math.IsNaN(float64(clipMaxLog)) || math.IsInf(float64(clipMaxLog), 0) {
		return nil, 0, fmt.Errorf("invalid Whisper clip mel maximum")
	}
	out, frames, _, err := whisperLogMel(ctx, samples, numMels, &clipMaxLog, false)
	return out, frames, err
}

// WhisperLogMelMaxContext returns one window's raw log10 mel maximum (-10 for
// digital silence) using WhisperLogMelContext's arithmetic.
func WhisperLogMelMaxContext(ctx context.Context, samples []float32, numMels int) (float32, error) {
	_, _, maxLog, err := whisperLogMel(ctx, samples, numMels, nil, true)
	return maxLog, err
}

func whisperLogMel(ctx context.Context, samples []float32, numMels int, clipMaxLog *float32, maxOnly bool) ([]float32, int, float32, error) {
	out, frames, maxLog, err := whisperLogMelRaw(ctx, samples, numMels)
	if err != nil || frames == 0 || maxOnly {
		return nil, frames, maxLog, err
	}
	if clipMaxLog != nil && *clipMaxLog > maxLog {
		maxLog = *clipMaxLog
	}
	floor := maxLog - 8
	for i, value := range out {
		if i%16384 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, 0, 0, err
			}
		}
		if value < floor {
			value = floor
		}
		out[i] = (value + 4) / 4
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, 0, err
	}
	return out, frames, maxLog, nil
}

// whisperLogMelRaw returns un-normalised log10 mel values and their maximum.
func whisperLogMelRaw(ctx context.Context, samples []float32, numMels int) ([]float32, int, float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, 0, 0, err
	}
	if numMels != 80 && numMels != 128 {
		return nil, 0, 0, fmt.Errorf("unsupported Whisper mel band count %d", numMels)
	}
	if len(samples) < whisperHop || len(samples) > 30*16000 {
		return nil, 0, 0, fmt.Errorf("Whisper window requires 160..480000 mono samples")
	}
	for index, value := range samples {
		if index%16384 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, 0, 0, err
			}
		}
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, 0, 0, fmt.Errorf("non-finite Whisper waveform")
		}
	}
	allFrames := 1 + len(samples)/whisperHop
	frames := allFrames - 1 // WhisperFeatureExtractor deliberately drops this.
	if frames <= 0 {
		return nil, 0, 0, nil
	}
	out := make([]float32, numMels*frames)
	nonzero := false
	for index, sample := range samples {
		if index%16384 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, 0, 0, err
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
					return nil, 0, 0, err
				}
			}
			out[i] = -10 // log10(1e-10)
		}
		if err := ctx.Err(); err != nil {
			return nil, 0, 0, err
		}
		return out, frames, -10, nil
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, 0, err
	}
	whisperExactTables.Do(initWhisperExactTables)
	if err := ctx.Err(); err != nil {
		return nil, 0, 0, err
	}
	filters := whisperExactTables.filters
	if numMels == 128 {
		filters = whisperExactTables.filters128
	}
	spans := whisperExactTables.spans
	if numMels == 128 {
		spans = whisperExactTables.spans128
	}
	centered := reflectCenter(samples, whisperFFTSize/2)
	// Frames are independent; each worker keeps the exact per-frame arithmetic
	// and its own scratch. The maximum is order-independent.
	workers := max(1, min(whisperMelWorkers, runtime.GOMAXPROCS(0), frames/64))
	chunk := (frames + workers - 1) / workers
	maxes := make([]float32, workers)
	errs := make([]error, workers)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo, hi := w*chunk, min((w+1)*chunk, frames)
		wg.Add(1)
		go func(w, lo, hi int) {
			defer wg.Done()
			maxes[w], errs[w] = whisperLogMelFrames(ctx, out, centered, filters, spans, numMels, frames, lo, hi)
		}(w, lo, hi)
	}
	wg.Wait()
	maxLog := float32(-math.MaxFloat32)
	for w := range maxes {
		if errs[w] != nil {
			return nil, 0, 0, errs[w]
		}
		if maxes[w] > maxLog {
			maxLog = maxes[w]
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, 0, err
	}
	return out, frames, maxLog, nil
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
	t.fft400 = newWhisperFFT400Plan()
	t.filters = whisperSlaneyFilters()
	t.filters128 = whisperSlaneyFiltersFor(128)
	t.spans = whisperFilterSpans(t.filters, 80)
	t.spans128 = whisperFilterSpans(t.filters128, 128)
}

// whisperFilterSpans returns each mel's nonzero bin range. Skipping exact-zero
// weights is bit-exact: the power spectrum is finite and non-negative, the sum
// starts at +0, and adding +0*power never changes it; retained terms keep
// their ascending-bin order.
func whisperFilterSpans(filters []float64, numMels int) [][2]int {
	spans := make([][2]int, numMels)
	for mel := range spans {
		lo, hi := whisperBins, 0
		for bin := 0; bin < whisperBins; bin++ {
			if filters[bin*numMels+mel] != 0 {
				lo, hi = min(lo, bin), bin+1
			}
		}
		if hi == 0 {
			lo = 0
		}
		spans[mel] = [2]int{lo, hi}
	}
	return spans
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

// whisperMelWorkers bounds mel frame parallelism (matching the CPU encoder's
// four-worker budget).
var whisperMelWorkers = 4

func whisperLogMelFrames(ctx context.Context, out []float32, centered []float32, filters []float64, spans [][2]int, numMels, frames, lo, hi int) (float32, error) {
	power := make([]float64, whisperBins)
	windowed := make([]float64, whisperFFTSize)
	fftScratch := make([]complex128, whisperFFTSize)
	maxLog := float32(-math.MaxFloat32)
	for frame := lo; frame < hi; frame++ {
		if (frame-lo)%64 == 0 {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
		}
		start := frame * whisperHop
		for sample := range windowed {
			windowed[sample] = float64(centered[start+sample]) * whisperExactTables.window[sample]
		}
		if !whisperExactTables.fft400.powerSpectrum400(power, windowed, fftScratch) {
			return 0, fmt.Errorf("Whisper FFT400 rejected fixed geometry")
		}
		for mel := 0; mel < numMels; mel++ {
			energy := float64(0)
			for bin := spans[mel][0]; bin < spans[mel][1]; bin++ {
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
	return maxLog, nil
}
