package whisper

import "testing"

func TestDecoderStateTracksLastToken(t *testing.T) {
	cfg := Tiny()
	cfg.DecoderLayers = 0
	dec := NewDecoder(cfg)
	encLen := 2
	encoderOutput := make([]float32, encLen*cfg.EncoderDModel)
	state := NewDecoderState(cfg, encoderOutput, encLen, dec)
	if state.LastToken != -1 {
		t.Fatalf("initial LastToken=%d want -1", state.LastToken)
	}
	if got := lastToken(state); got != TokenNoTimestamps {
		t.Fatalf("lastToken before prompt=%d want notimestamps", got)
	}
	dec.ForwardToken(TokenSOT, state)
	if state.LastToken != TokenSOT {
		t.Fatalf("LastToken=%d want SOT", state.LastToken)
	}
	dec.ForwardToken(TokenTranslate, state)
	if got := lastToken(state); got != TokenTranslate {
		t.Fatalf("lastToken=%d want translate", got)
	}
}

func TestSpeculativeDecodePromptFeedsTranslatePrompt(t *testing.T) {
	cfg := Tiny()
	cfg.DecoderLayers = 0
	cfg.MaxDecoderLength = 1
	draft := &Whisper{Decoder: NewDecoder(cfg), Config: cfg}
	target := &Whisper{Decoder: NewDecoder(cfg), Config: cfg}
	encLen := 1
	enc := make([]float32, encLen*cfg.EncoderDModel)
	out := (&SpeculativeDecoder{Draft: draft, Target: target, K: 1}).SpeculativeDecodePrompt(enc, encLen, LanguageTokens["es"], TokenTranslate)
	if len(out) == 0 {
		t.Fatalf("empty speculative output")
	}
	if target.Decoder == nil {
		t.Fatal("target decoder nil")
	}
}

func TestDecoderStateCheckpointRestore(t *testing.T) {
	cfg := Tiny()
	cfg.DecoderLayers = 0
	dec := NewDecoder(cfg)
	state := NewDecoderState(cfg, make([]float32, cfg.EncoderDModel), 1, dec)
	dec.ForwardToken(TokenSOT, state)
	cp := checkpointDecoderState(state)
	dec.ForwardToken(TokenEnglish, state)
	dec.ForwardToken(TokenTranslate, state)
	restoreDecoderState(state, cp)
	if state.Pos != 1 || state.LastToken != TokenSOT {
		t.Fatalf("restored pos=%d last=%d want pos=1 last=SOT", state.Pos, state.LastToken)
	}
}

func TestSpeculativeStepReplaysDraftStateToEmittedPrefix(t *testing.T) {
	cfg := Tiny()
	cfg.DecoderLayers = 0
	cfg.MaxDecoderLength = 8
	draft := &Whisper{Decoder: NewDecoder(cfg), Config: cfg}
	target := &Whisper{Decoder: NewDecoder(cfg), Config: cfg}
	encLen := 1
	enc := make([]float32, encLen*cfg.EncoderDModel)
	draftState := NewDecoderState(cfg, enc, encLen, draft.Decoder)
	targetState := NewDecoderState(cfg, enc, encLen, target.Decoder)
	for _, tok := range []int{TokenSOT, TokenEnglish, TokenTranslate, TokenNoTimestamps} {
		draft.Decoder.ForwardToken(tok, draftState)
		target.Decoder.ForwardToken(tok, targetState)
	}
	res := (&SpeculativeDecoder{Draft: draft, Target: target, K: 3}).Step(draftState, targetState)
	if res.Bonus < 0 {
		t.Fatalf("invalid bonus token %d", res.Bonus)
	}
	if draftState.Pos != targetState.Pos || draftState.LastToken != targetState.LastToken {
		t.Fatalf("draft state diverged pos=%d last=%d target pos=%d last=%d", draftState.Pos, draftState.LastToken, targetState.Pos, targetState.LastToken)
	}
	for l := range targetState.SelfKCache {
		if len(draftState.SelfKCache[l]) != len(targetState.SelfKCache[l]) || len(draftState.SelfVCache[l]) != len(targetState.SelfVCache[l]) {
			t.Fatalf("layer %d draft KV lens k=%d v=%d target k=%d v=%d", l, len(draftState.SelfKCache[l]), len(draftState.SelfVCache[l]), len(targetState.SelfKCache[l]), len(targetState.SelfVCache[l]))
		}
	}
}

func TestSpeculativeStepUsesTrackedLastToken(t *testing.T) {
	cfg := Tiny()
	cfg.DecoderLayers = 0
	cfg.MaxDecoderLength = 8
	draft := &Whisper{Decoder: NewDecoder(cfg), Config: cfg}
	target := &Whisper{Decoder: NewDecoder(cfg), Config: cfg}
	encLen := 1
	enc := make([]float32, encLen*cfg.EncoderDModel)
	draftState := NewDecoderState(cfg, enc, encLen, draft.Decoder)
	targetState := NewDecoderState(cfg, enc, encLen, target.Decoder)
	for _, tok := range []int{TokenSOT, TokenEnglish, TokenTranslate, TokenNoTimestamps} {
		draft.Decoder.ForwardToken(tok, draftState)
		target.Decoder.ForwardToken(tok, targetState)
	}
	res := (&SpeculativeDecoder{Draft: draft, Target: target, K: 2}).Step(draftState, targetState)
	if res.Bonus < 0 {
		t.Fatalf("invalid bonus token %d", res.Bonus)
	}
	if targetState.LastToken < 0 {
		t.Fatalf("target LastToken was not advanced")
	}
}
