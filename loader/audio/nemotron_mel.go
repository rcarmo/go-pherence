package audio

import (
	"fmt"
	"math"
	"sync"

	"github.com/rcarmo/go-pherence/backends/simd/fft"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// NemotronLogMel computes the offline, centered 16 kHz frontend shared by the
// pinned Nemotron 3.5 ASR and Nemotron 3 Diarization processors. Results are
// frame-major [floor(len(samples)/160)+1, 128]; the last row is masked to zero.
// This is a frontend only: it neither runs a model nor handles streaming state.
func NemotronLogMel(samples []float32) ([]float32, int, error) {
	return nemotronLogMel(samples, true)
}

// NemotronLogMelScalar uses the same frontend with a scalar mel projection.
// It is a portable dispatch reference for the checked SIMD projection; the
// PyTorch-generated fixture supplies the independent numerical reference.
func NemotronLogMelScalar(samples []float32) ([]float32, int, error) {
	return nemotronLogMel(samples, false)
}

func nemotronLogMel(samples []float32, vector bool) ([]float32, int, error) {
	const hop, nfft, win, bins, mels = 160, 512, 400, 257, 128
	if len(samples) == 0 || len(samples) > 16000*30 {
		return nil, 0, fmt.Errorf("Nemotron frontend requires 1..480000 mono 16 kHz samples")
	}
	for _, v := range samples {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, 0, fmt.Errorf("Nemotron frontend rejects non-finite audio")
		}
	}
	frames := len(samples)/hop + 1
	// Centered STFT with constant zero padding; the processor masks the last
	// frame (valid length floor(L/hop)) even though the STFT produces it.
	pre := make([]float32, len(samples))
	pre[0] = samples[0]
	for i := 1; i < len(pre); i++ {
		pre[i] = samples[i] - .97*samples[i-1]
	}
	window := make([]float32, win)
	for i := range window {
		window[i] = float32(.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(win-1))))
	}
	power := make([]float32, frames*bins)
	frameInput := make([]float32, nfft)
	fftReal, fftImag := make([]float64, nfft), make([]float64, nfft)
	spectrum := make([]float32, bins*2)
	for row := 0; row < frames-1; row++ {
		clear(frameInput)
		start := row*hop - nfft/2 + (nfft-win)/2
		for i := 0; i < win; i++ {
			index := start + i
			if index >= 0 && index < len(pre) {
				frameInput[(nfft-win)/2+i] = pre[index] * window[i]
			}
		}
		if !fft.ForwardRealInto(spectrum, frameInput, fftReal, fftImag) {
			return nil, 0, fmt.Errorf("Nemotron FFT rejected validated frame")
		}
		for bin := 0; bin < bins; bin++ {
			re, im := spectrum[2*bin], spectrum[2*bin+1]
			magnitude := float32(math.Sqrt(float64(re*re + im*im)))
			power[row*bins+bin] = magnitude * magnitude
		}
	}
	filters := nemotronFilters()
	out := make([]float32, frames*mels)
	if vector && frames > 1 && simd.HasSgemmAsm {
		// DenseNNTo is checked and uses AVX2/FMA or NEON where available.
		// The final masked row remains zero in the output.
		if !simd.DenseNNTo(out[:(frames-1)*mels], power[:(frames-1)*bins], filters, frames-1, mels, bins, 1, bins, mels, mels) {
			return nil, 0, fmt.Errorf("Nemotron mel projection rejected validated shape")
		}
	} else {
		for row := 0; row < frames-1; row++ {
			for mel := 0; mel < mels; mel++ {
				var sum float32
				for bin := 0; bin < bins; bin++ {
					sum += power[row*bins+bin] * filters[bin*mels+mel]
				}
				out[row*mels+mel] = sum
			}
		}
	}
	for i := range out[:(frames-1)*mels] {
		out[i] = float32(math.Log(float64(out[i] + float32(1.0/(1<<24)))))
	}
	return out, frames, nil
}

var nemotronFilters = sync.OnceValue(nemotronSlaneyFilters)

// librosa.filters.mel(sr=16000,n_fft=512,n_mels=128,norm="slaney")
// produces float32 weights from float64 Slaney band edges.
func nemotronSlaneyFilters() []float32 {
	const bins, mels = 257, 128
	hzToMel := func(hz float64) float64 {
		if hz < 1000 {
			return hz * 3 / 200
		}
		return 15 + math.Log(hz/1000)*(27/math.Log(6.4))
	}
	melToHz := func(mel float64) float64 {
		if mel < 15 {
			return mel * 200 / 3
		}
		return 1000 * math.Exp((mel-15)*(math.Log(6.4)/27))
	}
	edges := make([]float64, mels+2)
	for i := range edges {
		edges[i] = melToHz(float64(i) * hzToMel(8000) / float64(mels+1))
	}
	filters := make([]float32, bins*mels)
	for bin := 0; bin < bins; bin++ {
		hz := float64(bin) * 16000 / 512
		for mel := 0; mel < mels; mel++ {
			left := (hz - edges[mel]) / (edges[mel+1] - edges[mel])
			right := (edges[mel+2] - hz) / (edges[mel+2] - edges[mel+1])
			weight := math.Max(0, math.Min(left, right))
			filters[bin*mels+mel] = float32(weight * 2 / (edges[mel+2] - edges[mel]))
		}
	}
	return filters
}
