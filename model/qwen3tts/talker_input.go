package qwen3tts

import "fmt"

// TalkerInputLayout captures the embedding-fusion contract for Talker prefill.
// Text tokens are projected from text_hidden into talker_hidden while codec
// control tokens are embedded in talker_hidden and overlaid at fixed positions.
type TalkerInputLayout struct {
	TextHiddenSize   int `json:"text_hidden_size"`
	TalkerHiddenSize int `json:"talker_hidden_size"`
	TextTokens       int `json:"text_tokens"`
	CodecTokens      int `json:"codec_tokens"`
	OverlayPosition  int `json:"overlay_position"`
	ProjectionFloats int `json:"projection_floats"`
	FusedInputFloats int `json:"fused_input_floats"`
}

func NewTalkerInputLayout(cfg ParsedConfig, prefill PrefillLayout) (TalkerInputLayout, error) {
	layout := TalkerInputLayout{
		TextHiddenSize:   cfg.TalkerTextHiddenSize,
		TalkerHiddenSize: cfg.TalkerHiddenSize,
		TextTokens:       prefill.TextTokens,
		CodecTokens:      prefill.CodecTokens,
		OverlayPosition:  prefill.OverlayPosition,
	}
	layout.ProjectionFloats = layout.TextHiddenSize * layout.TalkerHiddenSize
	layout.FusedInputFloats = layout.TextTokens * layout.TalkerHiddenSize
	return layout, layout.Validate()
}

func (l TalkerInputLayout) Validate() error {
	if l.TextHiddenSize <= 0 || l.TalkerHiddenSize <= 0 || l.TextTokens <= CustomVoiceFirstTextIndex || l.CodecTokens <= 0 {
		return fmt.Errorf("invalid Qwen3-TTS talker input layout dims: %+v", l)
	}
	if l.OverlayPosition != CustomVoiceFirstTextIndex {
		return fmt.Errorf("invalid Qwen3-TTS talker overlay position=%d want=%d", l.OverlayPosition, CustomVoiceFirstTextIndex)
	}
	if l.ProjectionFloats != l.TextHiddenSize*l.TalkerHiddenSize {
		return fmt.Errorf("invalid Qwen3-TTS text projection floats=%d want=%d", l.ProjectionFloats, l.TextHiddenSize*l.TalkerHiddenSize)
	}
	if l.FusedInputFloats != l.TextTokens*l.TalkerHiddenSize {
		return fmt.Errorf("invalid Qwen3-TTS fused input floats=%d want=%d", l.FusedInputFloats, l.TextTokens*l.TalkerHiddenSize)
	}
	return nil
}
