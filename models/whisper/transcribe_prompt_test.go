package whisper

import "testing"

func TestGreedyDecodePromptAcceptsTranslatePrompt(t *testing.T) {
	cfg := Tiny()
	cfg.DecoderLayers = 0
	cfg.MaxDecoderLength = 2
	dec := NewDecoder(cfg)
	dModel := cfg.DecoderDModel
	dec.TokenEmbed = make([]float32, cfg.VocabSize*dModel)
	dec.PosEmbed = make([]float32, cfg.MaxDecoderLength*dModel)
	dec.FinalLNWeight = ones(dModel)
	dec.FinalLNBias = make([]float32, dModel)
	state := NewDecoderState(cfg, make([]float32, cfg.EncoderDModel), 1, dec)
	out := GreedyDecodePrompt(dec, state, cfg, LanguageTokens["en"], TokenTranslate)
	if len(out) == 0 {
		t.Fatal("expected at least one token from prompt-aware decode")
	}
}
