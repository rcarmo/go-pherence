package qwen3tts

import "fmt"

// PrefillLayout describes the Talker prefill stream lengths produced from a
// tokenized Qwen3-TTS prompt. It keeps text and codec/control streams explicit
// because runtime embeddings overlay the codec stream onto selected positions.
type PrefillLayout struct {
	TextTokens       int `json:"text_tokens"`
	CodecTokens      int `json:"codec_tokens"`
	FirstTextIndex   int `json:"first_text_index"`
	OverlayPosition  int `json:"overlay_position"`
	TalkerHiddenSize int `json:"talker_hidden_size"`
	EmbeddingFloats  int `json:"embedding_floats"`
}

func NewPrefillLayout(cfg ParsedConfig, prompt PromptIDs) (PrefillLayout, error) {
	layout := PrefillLayout{
		TextTokens:       len(prompt.Text),
		CodecTokens:      len(prompt.Codec),
		FirstTextIndex:   CustomVoiceFirstTextIndex,
		OverlayPosition:  CustomVoiceFirstTextIndex,
		TalkerHiddenSize: cfg.TalkerHiddenSize,
	}
	layout.EmbeddingFloats = layout.TextTokens * layout.TalkerHiddenSize
	return layout, layout.Validate()
}

func (l PrefillLayout) Validate() error {
	if l.TextTokens <= CustomVoiceFirstTextIndex || l.CodecTokens <= 0 || l.TalkerHiddenSize <= 0 {
		return fmt.Errorf("invalid Qwen3-TTS prefill layout dims: %+v", l)
	}
	if l.FirstTextIndex != CustomVoiceFirstTextIndex || l.OverlayPosition != CustomVoiceFirstTextIndex {
		return fmt.Errorf("invalid Qwen3-TTS prefill first/overlay indices: %+v", l)
	}
	if l.EmbeddingFloats != l.TextTokens*l.TalkerHiddenSize {
		return fmt.Errorf("invalid Qwen3-TTS prefill embedding_floats=%d want=%d", l.EmbeddingFloats, l.TextTokens*l.TalkerHiddenSize)
	}
	return nil
}

func (l PrefillLayout) EmbeddingBytes(bytesPerFloat int) (int64, error) {
	if err := l.Validate(); err != nil {
		return 0, err
	}
	if bytesPerFloat <= 0 {
		return 0, fmt.Errorf("invalid prefill bytes/float=%d", bytesPerFloat)
	}
	return int64(l.EmbeddingFloats) * int64(bytesPerFloat), nil
}
