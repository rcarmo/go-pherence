package audio

import (
	"context"
	"fmt"
	"math"
	"sync"

	"github.com/rcarmo/go-pherence/backends/simd/fft"
)

const (
	WeSpeakerSampleRate    = 16000
	WeSpeakerMelBands      = 80
	WeSpeakerWindowSamples = 400
	WeSpeakerHopSamples    = 160
	WeSpeakerMaxSamples    = 160000
	weSpeakerFFT           = 512
	weSpeakerEpsilon       = float32(1.1920928955078125e-7)
)

var weSpeakerTables struct {
	sync.Once
	window [400]float32
	mel    [80 * 257]float32
}

// WeSpeakerFbank implements the fixed pyannote WeSpeaker default frontend for
// mono16k normalised float32 PCM. It scales by32768, removes DC independently
// per400-sample frame, applies0.97 pre-emphasis (replicated first sample), then
// nonperiodic Hamming and right-zero-padding to FFT512. Mel triangles span20Hz
// to8kHz, use power/natural-log with float32 epsilon, no energy column/dither/
// VTLN, followed by per-band whole-window mean centering.
//
// Only complete frames are included (snip_edges=true); output is owned
// frame-major [1+(len(samples)-400)/160,80]. Inputs400..160000 must be finite and
// immutable for the call. It does not resample, pad audio, load model weights or
// accept alternative checkpoint frontend settings. Mean centering couples all
// frames in the window; splitting a window changes the result. This is not the
// Whisper/SpeechBrain frontend and does not alter either existing API.
//
// Cancellation checks surround every frame/operator and bounded reduction blocks;
// FFT512 and one-time fixed table construction are synchronous. The existing
// fft.ForwardRealSIMD currently dispatches a Go float64 radix2 implementation;
// no new assembly/high-performance or real-model quality claim is made here.
func WeSpeakerFbank(ctx context.Context, samples []float32) ([]float32, int, error) {
	return WeSpeakerFbankObserved(ctx, samples, nil)
}

// WeSpeakerFbankObserver sees transient read-only values. Stages: "window"
// (frame index,512values), "power" (frame,257), "log_mel" and "centered"
// (frame=-1,frames*80). Copy to retain. Callbacks are synchronous; previously
// observed stages remain visible if a later stage is cancelled or fails.
type WeSpeakerFbankObserver func(stage string, frame int, values []float32)

func WeSpeakerFbankObserved(ctx context.Context, samples []float32, observe WeSpeakerFbankObserver) ([]float32, int, error) {
	fail := func(err error) ([]float32, int, error) { return nil, 0, err }
	if err := ctx.Err(); err != nil {
		return fail(err)
	}
	if len(samples) < WeSpeakerWindowSamples || len(samples) > WeSpeakerMaxSamples {
		return fail(fmt.Errorf("WeSpeaker frontend requires400..160000 mono16k samples"))
	}
	// Scaling finite normalised floats may still overflow for pathological callers.
	for i, v := range samples {
		if i%4096 == 0 {
			if err := ctx.Err(); err != nil {
				return fail(err)
			}
		}
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v)) > float64(math.MaxFloat32)/32768 {
			return fail(fmt.Errorf("invalid WeSpeaker PCM value"))
		}
	}
	weSpeakerTables.Do(initWeSpeakerTables)
	if err := ctx.Err(); err != nil {
		return fail(err)
	}
	frames := 1 + (len(samples)-400)/160
	out := make([]float32, frames*80)
	var window [512]float32
	var power [257]float32
	for frame := 0; frame < frames; frame++ {
		if err := ctx.Err(); err != nil {
			return fail(err)
		}
		for i, v := range samples[frame*160 : frame*160+400] {
			window[i] = v * 32768
		}
		mu := weSpeakerMean400(window[:400])
		for i := 0; i < 400; i++ {
			window[i] -= mu
		}
		// Traverse backward so pre-emphasis never uses an already transformed sample.
		for i := 399; i >= 0; i-- {
			previous := window[max(0, i-1)]
			window[i] = (window[i] - float32(0.97)*previous) * weSpeakerTables.window[i]
		}
		if err := finiteWeSpeaker(ctx, window[:]); err != nil {
			return fail(err)
		}
		if observe != nil {
			observe("window", frame, window[:])
		}
		if err := ctx.Err(); err != nil {
			return fail(err)
		}
		spectrum := fft.ForwardRealSIMD(window[:])
		if len(spectrum) != 514 {
			return fail(fmt.Errorf("WeSpeaker FFT shape failure"))
		}
		for bin := range power {
			magnitude := float32(math.Hypot(float64(spectrum[bin*2]), float64(spectrum[bin*2+1])))
			power[bin] = magnitude * magnitude
		}
		if err := finiteWeSpeaker(ctx, power[:]); err != nil {
			return fail(err)
		}
		if observe != nil {
			observe("power", frame, power[:])
		}
		if err := ctx.Err(); err != nil {
			return fail(err)
		}
		for mel := 0; mel < 80; mel++ {
			// Float32 products with float64 sum keep bounded reduction error. The
			// reference uses float32 matrix multiplication, compared by fixture gates.
			energy := float64(0)
			for bin, p := range power {
				energy += float64(p * weSpeakerTables.mel[mel*257+bin])
			}
			e := float32(energy)
			if math.IsNaN(float64(e)) || math.IsInf(float64(e), 0) {
				return fail(fmt.Errorf("non-finite WeSpeaker mel energy"))
			}
			out[frame*80+mel] = float32(math.Log(float64(max(e, weSpeakerEpsilon))))
		}
	}
	if observe != nil {
		observe("log_mel", -1, out)
	}
	if err := ctx.Err(); err != nil {
		return fail(err)
	}
	for mel := 0; mel < 80; mel++ {
		if err := ctx.Err(); err != nil {
			return fail(err)
		}
		var sum float64
		for frame := 0; frame < frames; frame++ {
			sum += float64(out[frame*80+mel])
		}
		mean := float32(sum / float64(frames))
		for frame := 0; frame < frames; frame++ {
			out[frame*80+mel] -= mean
		}
	}
	if err := finiteWeSpeaker(ctx, out); err != nil {
		return fail(err)
	}
	if observe != nil {
		observe("centered", -1, out)
	}
	if err := ctx.Err(); err != nil {
		return fail(err)
	}
	return out, frames, nil
}

func initWeSpeakerTables() {
	for i := range weSpeakerTables.window {
		// Torch nonperiodic Hamming uses float32 angle and cosine arithmetic.
		angle := float32(i) * float32(2*math.Pi/399)
		weSpeakerTables.window[i] = float32(0.54) - float32(0.46)*float32(math.Cos(float64(angle)))
	}
	low := 1127 * math.Log(1+20.0/700)
	high := 1127 * math.Log(1+8000.0/700)
	delta := float32((high - low) / 81)
	lo := float32(low)
	for band := 0; band < 80; band++ {
		left := lo + float32(band)*delta
		center := lo + float32(band+1)*delta
		right := lo + float32(band+2)*delta
		for bin := 0; bin < 256; bin++ {
			// Mirrors tensor mel_scale: float32 ratio, add, log and multiply.
			ratio := float32(float32(bin)*float32(16000.0/512)) / 700
			mel := float32(1127) * float32(math.Log(float64(float32(1)+ratio)))
			weight := max(float32(0), min((mel-left)/(center-left), (right-mel)/(right-center)))
			weSpeakerTables.mel[band*257+bin] = weight
		}
		// Nyquist column deliberately remains zero as in kaldi.fbank padding.
	}
}
func finiteWeSpeaker(ctx context.Context, values []float32) error {
	for i, v := range values {
		if i%4096 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("non-finite WeSpeaker frontend stage")
		}
	}
	return ctx.Err()
}

// Fixed400-element float32 reduction matching the pinned Torch AVX2 inner sum:
// four interleaved eight-lane partials, then the two remaining vectors, combine
// partials in order and finally lanes in order. This models numeric ordering in
// Go, not SIMD acceleration. Other Torch CPU architectures may reduce differently.
func weSpeakerMean400(x []float32) float32 {
	var partial [4][8]float32
	for block := 0; block < 12; block++ {
		for p := 0; p < 4; p++ {
			for lane := 0; lane < 8; lane++ {
				partial[p][lane] += x[block*32+p*8+lane]
			}
		}
	}
	for vector := 48; vector < 50; vector++ {
		for lane := 0; lane < 8; lane++ {
			partial[0][lane] += x[vector*8+lane]
		}
	}
	for p := 1; p < 4; p++ {
		for lane := 0; lane < 8; lane++ {
			partial[0][lane] += partial[p][lane]
		}
	}
	var sum float32
	for _, value := range partial[0] {
		sum += value
	}
	return sum / 400
}
