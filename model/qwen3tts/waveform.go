package qwen3tts

import "fmt"

const (
	decoder12HzSampleRate       = 24000
	decoder12HzSamplesPerFrame  = 1920
	decoder12HzNominalFrameRate = 12
)

// WaveformLayout captures Decoder12Hz's exact tensor geometry. The published
// stage name is nominally 12Hz, but the neural topology upsamples every codec
// frame by 2*2*8*5*4*3 = 1920 samples. At 24kHz that is exactly 12.5 frames/s;
// sizing must follow topology rather than integer label division.
type WaveformLayout struct {
	FrameRateHz     int `json:"frame_rate_hz"`
	SampleRateHz    int `json:"sample_rate_hz"`
	Channels        int `json:"channels"`
	SamplesPerFrame int `json:"samples_per_frame"`
}

func NewWaveformLayout(dec DecoderPlan) (WaveformLayout, error) {
	layout := WaveformLayout{FrameRateHz: dec.FrameRateHz, SampleRateHz: decoder12HzSampleRate, Channels: 1, SamplesPerFrame: decoder12HzSamplesPerFrame}
	return layout, layout.Validate()
}

func (l WaveformLayout) Validate() error {
	if l.FrameRateHz != decoder12HzNominalFrameRate || l.SampleRateHz != decoder12HzSampleRate || l.Channels != 1 || l.SamplesPerFrame != decoder12HzSamplesPerFrame {
		return fmt.Errorf("invalid Qwen3-TTS waveform layout: %+v", l)
	}
	return nil
}

func (l WaveformLayout) SamplesForFrames(frames int) (int, error) {
	if err := l.Validate(); err != nil {
		return 0, err
	}
	if frames < 0 {
		return 0, fmt.Errorf("invalid Qwen3-TTS frame count=%d", frames)
	}
	return sizeCount(frames, l.SamplesPerFrame)
}

// FramesForSeconds returns a conservative frame ceiling at the exact topology
// cadence, so a duration limit never under-allocates decoder output.
func (l WaveformLayout) FramesForSeconds(seconds float64) (int, error) {
	if err := l.Validate(); err != nil {
		return 0, err
	}
	if seconds <= 0 {
		return 0, fmt.Errorf("invalid Qwen3-TTS seconds=%g", seconds)
	}
	frames := seconds * float64(l.SampleRateHz) / float64(l.SamplesPerFrame)
	maxInt := int(^uint(0) >> 1)
	if frames >= float64(maxInt) {
		return 0, fmt.Errorf("Qwen3-TTS seconds overflows frame count")
	}
	whole := int(frames)
	if float64(whole) < frames {
		whole++
	}
	return whole, nil
}
