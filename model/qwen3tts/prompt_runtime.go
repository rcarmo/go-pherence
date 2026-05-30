package qwen3tts

import "fmt"

// PromptRuntimeLayout ties tokenized prompt streams to the Talker input-fusion
// contract. It is prompt-specific, so it lives outside RuntimePlan, but inspector
// and runtime prefill code should use it whenever concrete PromptIDs are known.
type PromptRuntimeLayout struct {
	Prefill     PrefillLayout     `json:"prefill"`
	TalkerInput TalkerInputLayout `json:"talker_input"`
}

func NewPromptRuntimeLayout(cfg ParsedConfig, prompt PromptIDs) (PromptRuntimeLayout, error) {
	prefill, err := NewPrefillLayout(cfg, prompt)
	if err != nil {
		return PromptRuntimeLayout{}, err
	}
	talkerInput, err := NewTalkerInputLayout(cfg, prefill)
	if err != nil {
		return PromptRuntimeLayout{}, err
	}
	layout := PromptRuntimeLayout{Prefill: prefill, TalkerInput: talkerInput}
	return layout, layout.Validate()
}

func (l PromptRuntimeLayout) Validate() error {
	if err := l.Prefill.Validate(); err != nil {
		return err
	}
	if err := l.TalkerInput.Validate(); err != nil {
		return err
	}
	if l.Prefill.TextTokens != l.TalkerInput.TextTokens || l.Prefill.CodecTokens != l.TalkerInput.CodecTokens {
		return fmt.Errorf("Qwen3-TTS prompt prefill/talker token mismatch: prefill=%+v talker_input=%+v", l.Prefill, l.TalkerInput)
	}
	if l.Prefill.TalkerHiddenSize != l.TalkerInput.TalkerHiddenSize || l.Prefill.EmbeddingFloats != l.TalkerInput.FusedInputFloats {
		return fmt.Errorf("Qwen3-TTS prompt prefill/talker hidden mismatch: prefill=%+v talker_input=%+v", l.Prefill, l.TalkerInput)
	}
	if l.Prefill.OverlayPosition != l.TalkerInput.OverlayPosition {
		return fmt.Errorf("Qwen3-TTS prompt overlay mismatch: prefill=%d talker_input=%d", l.Prefill.OverlayPosition, l.TalkerInput.OverlayPosition)
	}
	return nil
}
