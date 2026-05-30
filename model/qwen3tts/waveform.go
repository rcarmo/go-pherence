package qwen3tts

import "fmt"

// WaveformLayout captures the Decoder12Hz output contract before waveform
// synthesis is implemented. Qwen3-TTS decoder frames are produced at 12Hz and
// expected to decode to mono 24kHz PCM/WAV samples.
type WaveformLayout struct {
	FrameRateHz     int `json:"frame_rate_hz"`
	SampleRateHz    int `json:"sample_rate_hz"`
	Channels        int `json:"channels"`
	SamplesPerFrame int `json:"samples_per_frame"`
}

func NewWaveformLayout(dec DecoderPlan) (WaveformLayout, error) {
	layout := WaveformLayout{FrameRateHz: dec.FrameRateHz, SampleRateHz: 24000, Channels: 1}
	if layout.FrameRateHz > 0 {
		layout.SamplesPerFrame = layout.SampleRateHz / layout.FrameRateHz
	}
	return layout, layout.Validate()
}

func (l WaveformLayout) Validate() error {
	if l.FrameRateHz <= 0 || l.SampleRateHz <= 0 || l.Channels <= 0 || l.SamplesPerFrame <= 0 {
		return fmt.Errorf("invalid Qwen3-TTS waveform layout: %+v", l)
	}
	if l.SampleRateHz%l.FrameRateHz != 0 {
		return fmt.Errorf("Qwen3-TTS sample_rate=%d is not divisible by frame_rate=%d", l.SampleRateHz, l.FrameRateHz)
	}
	if l.SamplesPerFrame != l.SampleRateHz/l.FrameRateHz {
		return fmt.Errorf("invalid Qwen3-TTS samples/frame=%d want=%d", l.SamplesPerFrame, l.SampleRateHz/l.FrameRateHz)
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
	return frames * l.SamplesPerFrame, nil
}
