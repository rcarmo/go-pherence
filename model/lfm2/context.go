package lfm2

import "fmt"

// ContextLayout captures token/context limits before generation is implemented.
// It keeps the 128k advertised context and tied-LM-head expectation visible in
// inspector/runtime-plan output.
type ContextLayout struct {
	VocabSize             int     `json:"vocab_size"`
	MaxPositionEmbeddings int     `json:"max_position_embeddings"`
	TieWordEmbeddings     bool    `json:"tie_word_embeddings"`
	RoPETheta             float64 `json:"rope_theta"`
}

func NewContextLayout(cfg Config) (ContextLayout, error) {
	if err := cfg.Validate(); err != nil {
		return ContextLayout{}, err
	}
	vocabSize := cfg.VocabSize
	if vocabSize == 0 {
		vocabSize = 128000
	}
	maxPos := cfg.MaxPositionEmbeddings
	if maxPos == 0 {
		maxPos = 128000
	}
	layout := ContextLayout{
		VocabSize:             vocabSize,
		MaxPositionEmbeddings: maxPos,
		TieWordEmbeddings:     cfg.TieWordEmbeddings,
		RoPETheta:             cfg.RoPE.Theta,
	}
	return layout, layout.Validate()
}

func (l ContextLayout) Validate() error {
	if l.VocabSize <= 0 || l.MaxPositionEmbeddings <= 0 {
		return fmt.Errorf("invalid LFM2 context layout: %+v", l)
	}
	if l.RoPETheta < 0 {
		return fmt.Errorf("invalid LFM2 rope theta=%g", l.RoPETheta)
	}
	return nil
}

func (l ContextLayout) ValidateToken(id uint32) error {
	if err := l.Validate(); err != nil {
		return err
	}
	if int(id) >= l.VocabSize {
		return fmt.Errorf("invalid LFM2 token id=%d vocab=%d", id, l.VocabSize)
	}
	return nil
}

func (l ContextLayout) ValidateSequence(ids []uint32) error {
	if len(ids) == 0 {
		return fmt.Errorf("empty LFM2 token sequence")
	}
	if len(ids) > l.MaxPositionEmbeddings {
		return fmt.Errorf("LFM2 token sequence length=%d exceeds max_position_embeddings=%d", len(ids), l.MaxPositionEmbeddings)
	}
	for i, id := range ids {
		if err := l.ValidateToken(id); err != nil {
			return fmt.Errorf("token[%d]: %w", i, err)
		}
	}
	return nil
}
