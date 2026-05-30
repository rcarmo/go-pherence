package qwen3tts

import "fmt"

// DecoderInputLayout captures the acoustic-code tensor passed from the
// CodePredictor to the 12Hz decoder. It is distinct from AcousticFrameLayout:
// frames are grouped as acoustic codebooks only, with semantic group 0 stripped
// before waveform decoding.
type DecoderInputLayout struct {
	FrameRateHz    int `json:"frame_rate_hz"`
	AcousticGroups int `json:"acoustic_groups"`
	CodecVocab     int `json:"codec_vocab"`
	CodesPerFrame  int `json:"codes_per_frame"`
	SemanticGroup  int `json:"semantic_group"`
	FirstCodeGroup int `json:"first_code_group"`
	LastCodeGroup  int `json:"last_code_group"`
}

func NewDecoderInputLayout(cfg ParsedConfig) (DecoderInputLayout, error) {
	if err := cfg.Validate(); err != nil {
		return DecoderInputLayout{}, err
	}
	layout := DecoderInputLayout{
		FrameRateHz:    12,
		AcousticGroups: cfg.CPNumCodeGroups - 1,
		CodecVocab:     cfg.CPVocabSize,
		CodesPerFrame:  cfg.CPNumCodeGroups - 1,
		SemanticGroup:  0,
		FirstCodeGroup: 1,
		LastCodeGroup:  cfg.CPNumCodeGroups - 1,
	}
	return layout, layout.Validate()
}

func (l DecoderInputLayout) Validate() error {
	if l.FrameRateHz != 12 || l.AcousticGroups <= 0 || l.CodecVocab <= 0 || l.SemanticGroup != 0 || l.FirstCodeGroup != 1 {
		return fmt.Errorf("invalid Qwen3-TTS decoder input layout: %+v", l)
	}
	if l.CodesPerFrame != l.AcousticGroups {
		return fmt.Errorf("invalid Qwen3-TTS decoder codes/frame=%d want=%d", l.CodesPerFrame, l.AcousticGroups)
	}
	if l.LastCodeGroup != l.FirstCodeGroup+l.AcousticGroups-1 {
		return fmt.Errorf("invalid Qwen3-TTS decoder last code group=%d want=%d", l.LastCodeGroup, l.FirstCodeGroup+l.AcousticGroups-1)
	}
	return nil
}

func (l DecoderInputLayout) CodesForFrames(frames int) (int, error) {
	if err := l.Validate(); err != nil {
		return 0, err
	}
	if frames < 0 {
		return 0, fmt.Errorf("invalid Qwen3-TTS decoder frame count=%d", frames)
	}
	return frames * l.CodesPerFrame, nil
}

func (l DecoderInputLayout) ValidateCodes(codes []uint32) error {
	if err := l.Validate(); err != nil {
		return err
	}
	if len(codes)%l.CodesPerFrame != 0 {
		return fmt.Errorf("invalid Qwen3-TTS decoder code count=%d not divisible by codes/frame=%d", len(codes), l.CodesPerFrame)
	}
	for i, code := range codes {
		if int(code) >= l.CodecVocab {
			return fmt.Errorf("Qwen3-TTS decoder code[%d]=%d exceeds codec vocab=%d", i, code, l.CodecVocab)
		}
	}
	return nil
}
