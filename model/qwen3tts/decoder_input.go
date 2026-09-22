package qwen3tts

import "fmt"

// DecoderInputLayout captures the complete codec-frame tensor passed to
// Decoder12Hz. CodePredictor emits only groups 1..15; callers must join the
// Talker semantic token as group 0 before waveform decoding.
type DecoderInputLayout struct {
	FrameRateHz     int `json:"frame_rate_hz"`
	TotalCodeGroups int `json:"total_code_groups"`
	AcousticGroups  int `json:"acoustic_groups"`
	CodecVocab      int `json:"codec_vocab"`
	CodesPerFrame   int `json:"codes_per_frame"`
	SemanticGroup   int `json:"semantic_group"`
	FirstCodeGroup  int `json:"first_code_group"`
	LastCodeGroup   int `json:"last_code_group"`
}

func NewDecoderInputLayout(cfg ParsedConfig) (DecoderInputLayout, error) {
	if err := cfg.Validate(); err != nil {
		return DecoderInputLayout{}, err
	}
	layout := DecoderInputLayout{
		FrameRateHz:     12,
		TotalCodeGroups: cfg.CPNumCodeGroups,
		AcousticGroups:  cfg.CPNumCodeGroups - 1,
		CodecVocab:      cfg.CPVocabSize,
		CodesPerFrame:   cfg.CPNumCodeGroups,
		SemanticGroup:   0,
		FirstCodeGroup:  1,
		LastCodeGroup:   cfg.CPNumCodeGroups - 1,
	}
	return layout, layout.Validate()
}

func (l DecoderInputLayout) Validate() error {
	if l.FrameRateHz != 12 || l.TotalCodeGroups < 2 || l.AcousticGroups <= 0 || l.CodecVocab <= 0 || l.SemanticGroup != 0 || l.FirstCodeGroup != 1 {
		return fmt.Errorf("invalid Qwen3-TTS decoder input layout: %+v", l)
	}
	if l.AcousticGroups != l.TotalCodeGroups-1 || l.CodesPerFrame != l.TotalCodeGroups {
		return fmt.Errorf("invalid Qwen3-TTS decoder groups total/acoustic/codes=%d/%d/%d", l.TotalCodeGroups, l.AcousticGroups, l.CodesPerFrame)
	}
	if l.LastCodeGroup != l.AcousticGroups {
		return fmt.Errorf("invalid Qwen3-TTS decoder last code group=%d want=%d", l.LastCodeGroup, l.AcousticGroups)
	}
	return nil
}

func (l DecoderInputLayout) DecoderPlan() (DecoderPlan, error) {
	if err := l.Validate(); err != nil {
		return DecoderPlan{}, err
	}
	return DecoderPlan{FrameRateHz: l.FrameRateHz, CodeGroups: l.TotalCodeGroups, CodesPerFrame: l.CodesPerFrame, CodecVocab: l.CodecVocab}, nil
}

// CodesForFrames returns the complete semantic+acoustic decoder code count.
func (l DecoderInputLayout) CodesForFrames(frames int) (int, error) {
	if err := l.Validate(); err != nil {
		return 0, err
	}
	if frames < 0 {
		return 0, fmt.Errorf("invalid Qwen3-TTS decoder frame count=%d", frames)
	}
	return sizeCount(frames, l.CodesPerFrame)
}

// AcousticCodesForFrames returns CodePredictor's groups 1..15 output count.
func (l DecoderInputLayout) AcousticCodesForFrames(frames int) (int, error) {
	if err := l.Validate(); err != nil {
		return 0, err
	}
	if frames < 0 {
		return 0, fmt.Errorf("invalid Qwen3-TTS decoder frame count=%d", frames)
	}
	return sizeCount(frames, l.AcousticGroups)
}

func (l DecoderInputLayout) ValidateCodes(codes []uint32) error {
	if err := l.Validate(); err != nil {
		return err
	}
	if len(codes) == 0 || len(codes)%l.CodesPerFrame != 0 {
		return fmt.Errorf("invalid Qwen3-TTS decoder code count=%d not divisible by codes/frame=%d", len(codes), l.CodesPerFrame)
	}
	for i, code := range codes {
		// The semantic vocabulary is 3072 and the tokenizer decoder maps group 0
		// modulo its 2048-entry codebook. Acoustic groups must already be <2048.
		if i%l.CodesPerFrame != l.SemanticGroup && int(code) >= l.CodecVocab {
			return fmt.Errorf("Qwen3-TTS decoder acoustic code[%d]=%d exceeds codec vocab=%d", i, code, l.CodecVocab)
		}
	}
	return nil
}

// JoinFrames interleaves semantic group 0 with flattened acoustic groups 1..15.
func (l DecoderInputLayout) JoinFrames(semantic, acoustic []uint32) ([]uint32, error) {
	if err := l.Validate(); err != nil {
		return nil, err
	}
	if len(semantic) == 0 {
		return nil, fmt.Errorf("Qwen3-TTS decoder semantic stream is empty")
	}
	wantAcoustic, err := l.AcousticCodesForFrames(len(semantic))
	if err != nil || len(acoustic) != wantAcoustic {
		return nil, fmt.Errorf("Qwen3-TTS decoder semantic/acoustic frame mismatch semantic=%d acoustic=%d want=%d", len(semantic), len(acoustic), wantAcoustic)
	}
	out := make([]uint32, len(semantic)*l.CodesPerFrame)
	for frame, code := range semantic {
		base := frame * l.CodesPerFrame
		out[base] = code
		copy(out[base+1:base+l.CodesPerFrame], acoustic[frame*l.AcousticGroups:(frame+1)*l.AcousticGroups])
	}
	if err := l.ValidateCodes(out); err != nil {
		return nil, err
	}
	return out, nil
}
