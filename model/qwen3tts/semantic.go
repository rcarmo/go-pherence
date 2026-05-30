package qwen3tts

import "fmt"

// SemanticTokenLayout captures the Talker -> CodePredictor handoff. Talker
// emits semantic codec-token IDs from the codec vocabulary; CodePredictor then
// consumes that stream as group 0 and predicts acoustic groups 1..N.
type SemanticTokenLayout struct {
	Group     int    `json:"group"`
	VocabSize int    `json:"vocab_size"`
	BOS       uint32 `json:"bos"`
	EOS       uint32 `json:"eos"`
	Pad       uint32 `json:"pad"`
	NoThink   uint32 `json:"no_think"`
	Think     uint32 `json:"think"`
	ThinkBOS  uint32 `json:"think_bos"`
	ThinkEOS  uint32 `json:"think_eos"`
}

func NewSemanticTokenLayout(cfg ParsedConfig) (SemanticTokenLayout, error) {
	vocab := cfg.TalkerVocabSize
	if vocab == 0 {
		vocab = CodecVocabSize
	}
	layout := SemanticTokenLayout{Group: 0, VocabSize: vocab, BOS: CodecBOS, EOS: CodecEOS, Pad: CodecPad, NoThink: CodecNoThink, Think: CodecThink, ThinkBOS: CodecThinkBOS, ThinkEOS: CodecThinkEOS}
	return layout, layout.Validate()
}

func (l SemanticTokenLayout) Validate() error {
	if l.Group != 0 || l.VocabSize <= 0 {
		return fmt.Errorf("invalid Qwen3-TTS semantic token layout: %+v", l)
	}
	for name, id := range map[string]uint32{"bos": l.BOS, "eos": l.EOS, "pad": l.Pad, "no_think": l.NoThink, "think": l.Think, "think_bos": l.ThinkBOS, "think_eos": l.ThinkEOS} {
		if int(id) >= l.VocabSize {
			return fmt.Errorf("invalid Qwen3-TTS semantic %s id=%d vocab=%d", name, id, l.VocabSize)
		}
	}
	return nil
}

func (l SemanticTokenLayout) ValidateToken(id uint32) error {
	if err := l.Validate(); err != nil {
		return err
	}
	if int(id) >= l.VocabSize {
		return fmt.Errorf("invalid Qwen3-TTS semantic token id=%d vocab=%d", id, l.VocabSize)
	}
	return nil
}

func (l SemanticTokenLayout) ValidateSequence(ids []uint32) error {
	if len(ids) == 0 {
		return fmt.Errorf("empty Qwen3-TTS semantic token sequence")
	}
	for i, id := range ids {
		if err := l.ValidateToken(id); err != nil {
			return fmt.Errorf("semantic token[%d]: %w", i, err)
		}
	}
	return nil
}
