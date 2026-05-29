package qwen3tts

import "fmt"

// AcousticFrameLayout describes the CodePredictor -> Decoder12Hz handoff. The
// first code group is the semantic token stream consumed by the predictor; the
// decoder receives the remaining acoustic code groups as one 12Hz frame.
type AcousticFrameLayout struct {
	TotalCodeGroups       int   `json:"total_code_groups"`
	SemanticGroup         int   `json:"semantic_group"`
	AcousticGroups        []int `json:"acoustic_groups"`
	AcousticCodesPerFrame int   `json:"acoustic_codes_per_frame"`
	CodecVocab            int   `json:"codec_vocab"`
}

func NewAcousticFrameLayout(cfg ParsedConfig) (AcousticFrameLayout, error) {
	if cfg.CPNumCodeGroups < 2 || cfg.CPVocabSize <= 0 {
		return AcousticFrameLayout{}, fmt.Errorf("invalid Qwen3-TTS acoustic frame config: code_groups=%d vocab=%d", cfg.CPNumCodeGroups, cfg.CPVocabSize)
	}
	groups := make([]int, 0, cfg.CPNumCodeGroups-1)
	for i := 1; i < cfg.CPNumCodeGroups; i++ {
		groups = append(groups, i)
	}
	layout := AcousticFrameLayout{TotalCodeGroups: cfg.CPNumCodeGroups, SemanticGroup: 0, AcousticGroups: groups, AcousticCodesPerFrame: len(groups), CodecVocab: cfg.CPVocabSize}
	return layout, layout.Validate()
}

func (l AcousticFrameLayout) Validate() error {
	if l.TotalCodeGroups < 2 || l.SemanticGroup != 0 || l.CodecVocab <= 0 {
		return fmt.Errorf("invalid Qwen3-TTS acoustic frame layout: %+v", l)
	}
	if len(l.AcousticGroups) != l.TotalCodeGroups-1 || l.AcousticCodesPerFrame != len(l.AcousticGroups) {
		return fmt.Errorf("invalid Qwen3-TTS acoustic group count: %+v", l)
	}
	for i, group := range l.AcousticGroups {
		want := i + 1
		if group != want {
			return fmt.Errorf("invalid Qwen3-TTS acoustic group at %d: got=%d want=%d", i, group, want)
		}
	}
	return nil
}

func (l AcousticFrameLayout) ValidateFrame(frame []uint32) error {
	if err := l.Validate(); err != nil {
		return err
	}
	if len(frame) != l.AcousticCodesPerFrame {
		return fmt.Errorf("invalid Qwen3-TTS acoustic frame length=%d want=%d", len(frame), l.AcousticCodesPerFrame)
	}
	for i, code := range frame {
		if int(code) >= l.CodecVocab {
			return fmt.Errorf("invalid Qwen3-TTS acoustic frame code[%d]=%d vocab=%d", i, code, l.CodecVocab)
		}
	}
	return nil
}
