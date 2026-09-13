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
	Hubert *Hubert
	Codec  *CodecEncoder
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

func (e *ReferenceEncoder) Encode(ctx context.Context, wave []float32, transcript string) (loader.CachedReferenceTokens, error) {
	if ctx == nil || e == nil || e.Hubert == nil || e.Codec == nil {
		return loader.CachedReferenceTokens{}, fmt.Errorf("omnivoice: invalid reference encoder/context")
	}
	if err := ctx.Err(); err != nil {
		return loader.CachedReferenceTokens{}, err
	}
	if len(wave) < 2*24000 || len(wave) > 20*24000 || strings.TrimSpace(transcript) == "" {
		return loader.CachedReferenceTokens{}, fmt.Errorf("omnivoice: reference requires 2..20 seconds at 24 kHz and transcript")
	}
	var sum float64
	for _, v := range wave {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return loader.CachedReferenceTokens{}, fmt.Errorf("omnivoice: nonfinite reference")
		}
		sum += float64(v) * float64(v)
	}
	rms := math.Sqrt(sum / float64(len(wave)))
	if rms < 1e-7 {
		return loader.CachedReferenceTokens{}, fmt.Errorf("omnivoice: silent reference")
	}
	// Upstream computes RMS before trimming to a codec hop boundary.
	normalized := append([]float32(nil), wave[:len(wave)/960*960]...)
	if rms < .1 {
		gain := float32(.1 / rms)
		for i := range normalized {
			normalized[i] *= gain
		}
	}
	semanticWave := resample24To16(normalized)
	padded := make([]float32, len(semanticWave)+320)
	copy(padded[160:], semanticWave)
	semantic, frames, err := e.Hubert.Extract(ctx, padded)
	if err != nil {
		return loader.CachedReferenceTokens{}, err
	}
	// Higgs config derives semantic_downsample_factor = 960 / 1.5 / 320 = 2.
	// Select every second HuBERT frame (no averaging).
	selectedFrames := (frames + 1) / 2
	for i := 0; i < selectedFrames; i++ {
		copy(semantic[i*768:(i+1)*768], semantic[i*2*768:(i*2+1)*768])
	}
	codes, err := e.Codec.EncodeFeatures(ctx, normalized, semantic[:selectedFrames*768], selectedFrames)
	if err != nil {
		return loader.CachedReferenceTokens{}, err
	}
	out := loader.CachedReferenceTokens{Books: 8, Frames: len(codes) / 8, Codes: codes, Transcript: transcript, RefRMS: &rms}
	return out, out.Validate()
}

// resample24To16 matches torchaudio's default Hann-windowed sinc polyphase
// resampler (width 6, rolloff .99), including float32 kernel arithmetic.
func resample24To16(wave []float32) []float32 {
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
	out := make([]float32, (len(wave)*2+2)/3)
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
	return out
}
