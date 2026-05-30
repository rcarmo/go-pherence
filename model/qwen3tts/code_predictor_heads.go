package qwen3tts

import "fmt"

// CodePredictorHeadLayout captures the acoustic-head contract. Group 0 is the
// semantic stream; the code predictor emits one head for each acoustic group.
type CodePredictorHeadLayout struct {
	SemanticGroup int   `json:"semantic_group"`
	HeadGroups    []int `json:"head_groups"`
	Heads         int   `json:"heads"`
	VocabSize     int   `json:"vocab_size"`
}

func NewCodePredictorHeadLayout(cfg ParsedConfig) (CodePredictorHeadLayout, error) {
	frame, err := NewAcousticFrameLayout(cfg)
	if err != nil {
		return CodePredictorHeadLayout{}, err
	}
	layout := CodePredictorHeadLayout{SemanticGroup: frame.SemanticGroup, HeadGroups: append([]int(nil), frame.AcousticGroups...), Heads: len(frame.AcousticGroups), VocabSize: cfg.CPVocabSize}
	return layout, layout.Validate()
}

func (l CodePredictorHeadLayout) Validate() error {
	if l.SemanticGroup != 0 || l.Heads <= 0 || l.VocabSize <= 0 || len(l.HeadGroups) != l.Heads {
		return fmt.Errorf("invalid Qwen3-TTS code predictor head layout: %+v", l)
	}
	for i, group := range l.HeadGroups {
		want := i + 1
		if group != want {
			return fmt.Errorf("invalid Qwen3-TTS code predictor head group at %d: got=%d want=%d", i, group, want)
		}
	}
	return nil
}

func (l CodePredictorHeadLayout) ValidateHeadLogits(head int, logits int) error {
	if err := l.Validate(); err != nil {
		return err
	}
	if head < 0 || head >= l.Heads {
		return fmt.Errorf("invalid Qwen3-TTS code predictor head=%d heads=%d", head, l.Heads)
	}
	if logits != l.VocabSize {
		return fmt.Errorf("invalid Qwen3-TTS code predictor logits=%d want=%d", logits, l.VocabSize)
	}
	return nil
}
