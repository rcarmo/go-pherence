package omnivoice

import (
	"context"
	"fmt"
	"math"
	"strings"

	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
)

// ReferenceEncoder combines HuBERT and DAC/RVQ. It is single-caller.
// Input is mono 24 kHz, with an explicitly supplied matching transcript.
// Silence trimming/ASR are intentionally not performed.
type ReferenceEncoder struct {
	Hubert                            *Hubert
	Codec                             *CodecEncoder
	normalized, padded, semantic      []float32
	samples, hubertFrames, codeFrames int
}

func LoadReferenceEncoder(path string) (*ReferenceEncoder, error) {
	hw, err := loader.LoadHubert(path)
	if err != nil {
		return nil, err
	}
	h, err := NewHubert(hw)
	if err != nil {
		return nil, err
	}
	cw, err := loader.LoadCodecEncoder(path)
	if err != nil {
		return nil, err
	}
	c, err := NewCodecEncoder(cw)
	if err != nil {
		return nil, err
	}
	return &ReferenceEncoder{Hubert: h, Codec: c}, nil
}

// Prepare reserves scratch for one exact reference length; returns codec frames.
func (e *ReferenceEncoder) Prepare(samples int) (int, error) {
	if e == nil || e.Hubert == nil || e.Codec == nil || samples < 48000 || samples > 480000 {
		return 0, fmt.Errorf("omnivoice: invalid reference encoder/length")
	}
	if e.samples == samples {
		return e.codeFrames, nil
	}
	normalized := samples / 960 * 960
	padded := normalized/3*2 + 320
	frames, err := e.Hubert.Prepare(padded)
	if err != nil {
		return 0, err
	}
	codeFrames := (frames + 1) / 2
	if err = e.Codec.Prepare(normalized, codeFrames); err != nil {
		return 0, err
	}
	e.normalized = reuseReferenceBuffer(e.normalized, normalized)
	e.padded = reuseReferenceBuffer(e.padded, padded)
	e.semantic = reuseReferenceBuffer(e.semantic, frames*768)
	e.samples = samples
	e.hubertFrames = frames
	e.codeFrames = codeFrames
	return codeFrames, nil
}

func reuseReferenceBuffer(buf []float32, n int) []float32 {
	if cap(buf) >= n {
		return buf[:n]
	}
	return make([]float32, n)
}

func (e *ReferenceEncoder) Encode(ctx context.Context, wave []float32, transcript string) (loader.CachedReferenceTokens, error) {
	if ctx == nil || strings.TrimSpace(transcript) == "" {
		return loader.CachedReferenceTokens{}, fmt.Errorf("omnivoice: context and transcript required")
	}
	if err := ctx.Err(); err != nil {
		return loader.CachedReferenceTokens{}, err
	}
	frames, err := e.Prepare(len(wave))
	if err != nil {
		return loader.CachedReferenceTokens{}, err
	}
	codes := make([]int, frames*8)
	rms, err := e.EncodeInto(ctx, codes, wave)
	if err != nil {
		return loader.CachedReferenceTokens{}, err
	}
	out := loader.CachedReferenceTokens{Books: 8, Frames: frames, Codes: codes, Transcript: transcript, RefRMS: &rms}
	return out, out.Validate()
}

// EncodeInto writes caller-owned codes; after Prepare successful calls allocate nothing.
// The input waveform is never modified. Returned RMS is measured before hop trimming.
func (e *ReferenceEncoder) EncodeInto(ctx context.Context, codes []int, wave []float32) (float64, error) {
	if ctx == nil || e == nil || e.Hubert == nil || e.Codec == nil {
		return 0, fmt.Errorf("omnivoice: invalid reference encoder/context")
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if e.samples != len(wave) || len(codes) != e.codeFrames*8 || e.samples == 0 {
		return 0, fmt.Errorf("omnivoice: reference scratch/output not prepared for input")
	}
	var sum float64
	for _, v := range wave {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return 0, fmt.Errorf("omnivoice: nonfinite reference")
		}
		sum += float64(v) * float64(v)
	}
	rms := math.Sqrt(sum / float64(len(wave)))
	if rms < 1e-7 {
		return 0, fmt.Errorf("omnivoice: silent reference")
	}
	// Upstream computes RMS before trimming to a codec hop boundary.
	normalized := e.normalized
	copy(normalized, wave[:len(normalized)])
	if rms < .1 {
		gain := float32(.1 / rms)
		for i := range normalized {
			normalized[i] *= gain
		}
	}
	clear(e.padded)
	resample24To16Into(e.padded[160:len(e.padded)-160], normalized)
	semantic, frames := e.semantic, e.hubertFrames
	if err := e.Hubert.ExtractInto(ctx, semantic, e.padded); err != nil {
		return 0, err
	}
	// Higgs config derives semantic_downsample_factor = 960 / 1.5 / 320 = 2.
	// Select every second HuBERT frame (no averaging).
	selectedFrames := (frames + 1) / 2
	for i := 0; i < selectedFrames; i++ {
		copy(semantic[i*768:(i+1)*768], semantic[i*2*768:(i*2+1)*768])
	}
	if err := e.Codec.EncodeFeaturesInto(ctx, codes, normalized, semantic[:selectedFrames*768], selectedFrames); err != nil {
		return 0, err
	}
	return rms, nil
}

// resample24To16 matches torchaudio's default Hann-windowed sinc polyphase
// resampler (width 6, rolloff .99), including float32 kernel arithmetic.
func resample24To16(wave []float32) []float32 {
	out := make([]float32, (len(wave)*2+2)/3)
	resample24To16Into(out, wave)
	return out
}

func resample24To16Into(out, wave []float32) {
	const width = 10
	var kernel [2][23]float32
	for phase := 0; phase < 2; phase++ {
		for j := -width; j < width+3; j++ {
			t := float32(float32(j)/3-float32(phase)/2) * float32(1.98)
			t = max(float32(-6), min(float32(6), t))
			angle := float32(float32(t*float32(math.Pi))/6) / 2
			win := float32(math.Cos(float64(angle)))
			win *= win
			x := t * float32(math.Pi)
			sinc := float32(1)
			if x != 0 {
				sinc = float32(math.Sin(float64(x))) / x
			}
			kernel[phase][j+width] = sinc * (win * float32(.66))
		}
	}
	for n := range out {
		base := n/2*3 - width
		var sum float32
		for j, k := range kernel[n%2] {
			i := base + j
			if i >= 0 && i < len(wave) {
				sum += wave[i] * k
			}
		}
		out[n] = sum
	}
}
