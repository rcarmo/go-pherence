package qwen3tts

import "fmt"

// SpeakerEncoderLayout captures the optional voice-reference encoder contract
// used by CustomVoice checkpoints. Runtime code should use this to validate the
// speaker embedding width and audio sample-rate assumptions before cloning or
// conditioning on reference speech.
type SpeakerEncoderLayout struct {
	Present           bool `json:"present"`
	EmbeddingDim      int  `json:"embedding_dim"`
	SampleRateHz      int  `json:"sample_rate_hz"`
	EmbeddingFloats   int  `json:"embedding_floats"`
	ReferenceChannels int  `json:"reference_channels"`
	SamplesPerSecond  int  `json:"samples_per_second"`
}

func NewSpeakerEncoderLayout(cfg ParsedConfig) (SpeakerEncoderLayout, error) {
	if err := cfg.Validate(); err != nil {
		return SpeakerEncoderLayout{}, err
	}
	if cfg.SpeakerEncoder == nil {
		return SpeakerEncoderLayout{Present: false}, nil
	}
	layout := SpeakerEncoderLayout{
		Present:           true,
		EmbeddingDim:      cfg.SpeakerEncoder.EncDim,
		SampleRateHz:      cfg.SpeakerEncoder.SampleRate,
		EmbeddingFloats:   cfg.SpeakerEncoder.EncDim,
		ReferenceChannels: 1,
		SamplesPerSecond:  cfg.SpeakerEncoder.SampleRate,
	}
	return layout, layout.Validate()
}

func (l SpeakerEncoderLayout) Validate() error {
	if !l.Present {
		if l.EmbeddingDim != 0 || l.SampleRateHz != 0 || l.EmbeddingFloats != 0 || l.ReferenceChannels != 0 || l.SamplesPerSecond != 0 {
			return fmt.Errorf("invalid absent Qwen3-TTS speaker encoder layout: %+v", l)
		}
		return nil
	}
	if l.EmbeddingDim <= 0 || l.SampleRateHz <= 0 || l.ReferenceChannels <= 0 {
		return fmt.Errorf("invalid Qwen3-TTS speaker encoder layout dims: %+v", l)
	}
	if l.EmbeddingFloats != l.EmbeddingDim {
		return fmt.Errorf("invalid Qwen3-TTS speaker embedding floats=%d want=%d", l.EmbeddingFloats, l.EmbeddingDim)
	}
	wantSamples := l.SampleRateHz * l.ReferenceChannels
	if l.SamplesPerSecond != wantSamples {
		return fmt.Errorf("invalid Qwen3-TTS speaker samples/second=%d want=%d", l.SamplesPerSecond, wantSamples)
	}
	return nil
}

func (l SpeakerEncoderLayout) ReferenceSamples(seconds int) (int, error) {
	if err := l.Validate(); err != nil {
		return 0, err
	}
	if !l.Present {
		return 0, fmt.Errorf("Qwen3-TTS speaker encoder is not present")
	}
	if seconds < 0 {
		return 0, fmt.Errorf("invalid Qwen3-TTS reference duration seconds=%d", seconds)
	}
	return seconds * l.SamplesPerSecond, nil
}
