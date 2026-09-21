package whisper

import "testing"

func TestForwardTokensMatchesForwardTokenLoop(t *testing.T) {
	cfg := Tiny()
	cfg.DecoderLayers = 0
	cfg.MaxDecoderLength = 8
	decA := NewDecoder(cfg)
	decB := NewDecoder(cfg)
	encLen := 1
	enc := make([]float32, encLen*cfg.EncoderDModel)
	stateA := NewDecoderState(cfg, enc, encLen, decA)
	stateB := NewDecoderState(cfg, enc, encLen, decB)
	tokens := []int{TokenSOT, TokenEnglish, TokenTranslate, TokenNoTimestamps}
	batched := decA.ForwardTokens(tokens, stateA)
	if len(batched) != len(tokens) {
		t.Fatalf("batched logits=%d want %d", len(batched), len(tokens))
	}
	for i, tok := range tokens {
		row := decB.ForwardToken(tok, stateB)
		if argmax(batched[i]) != argmax(row) {
			t.Fatalf("row %d argmax mismatch got %d want %d", i, argmax(batched[i]), argmax(row))
		}
	}
	if stateA.Pos != stateB.Pos || stateA.LastToken != stateB.LastToken {
		t.Fatalf("state mismatch batched pos=%d last=%d loop pos=%d last=%d", stateA.Pos, stateA.LastToken, stateB.Pos, stateB.LastToken)
	}
}

func TestVerifyDraftSequentialShape(t *testing.T) {
	cfg := Tiny()
	cfg.DecoderLayers = 0
	dec := NewDecoder(cfg)
	encLen := 1
	enc := make([]float32, encLen*cfg.EncoderDModel)
	state := NewDecoderState(cfg, enc, encLen, dec)
	logits, greedy := dec.VerifyDraftSequential(TokenNoTimestamps, []int{10, 11, 12}, state)
	if len(logits) != 4 || len(greedy) != 4 {
		t.Fatalf("shape logits=%d greedy=%d want 4", len(logits), len(greedy))
	}
	if state.LastToken != 12 {
		t.Fatalf("LastToken=%d want 12", state.LastToken)
	}
}
