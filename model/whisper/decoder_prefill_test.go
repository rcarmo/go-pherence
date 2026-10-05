package whisper

import (
	"math"
	"testing"
)

// Batched prompt prefill must leave the KV cache, position and every later
// logit bit-identical to feeding the same tokens one at a time.
func TestAdvanceTokensMatchesSequential(t *testing.T) {
	old := linearWorkers
	defer func() { linearWorkers = old }()
	for _, cfg := range []Config{
		{EncoderDModel: 64, DecoderDModel: 64, DecoderLayers: 2, DecoderHeads: 2, EncoderHeads: 2, HeadDim: 32, DecoderFFNDim: 128, EncoderFFNDim: 128, MaxDecoderLength: 40, VocabSize: 96},
		{EncoderDModel: 64, DecoderDModel: 64, DecoderLayers: 3, DecoderHeads: 4, EncoderHeads: 4, HeadDim: 16, DecoderFFNDim: 256, EncoderFFNDim: 256, MaxDecoderLength: 40, VocabSize: 96},
	} {
		frames := 11
		dec, layers, head, enc := newTestQ5Decoder(cfg, 21, frames)
		dec.q5Layers, dec.lmHeadQ5 = layers, head
		for _, workers := range []int{1, 3, 4} {
			linearWorkers = workers
			for _, compat := range []bool{false, true} {
				for _, n := range []int{2, 5, 23} {
					prompt := make([]int, n)
					for i := range prompt {
						prompt[i] = (i*37 + 5) % cfg.VocabSize
					}
					state := func() *DecoderState {
						s := NewDecoderState(cfg, enc, frames, dec)
						if compat {
							s.applyOriginalDecoderCompatibility(frames)
						}
						return s
					}
					seq, bat := state(), state()
					for _, tok := range prompt {
						dec.AdvanceToken(tok, seq)
					}
					if !dec.batchPrefillEligible(bat) {
						t.Fatal("batch prefill not exercised")
					}
					dec.AdvanceTokens(prompt, bat)
					if seq.Pos != bat.Pos || seq.LastToken != bat.LastToken {
						t.Fatal("position", seq.Pos, bat.Pos, seq.LastToken, bat.LastToken)
					}
					for l := range seq.SelfKCache {
						bitsEqual(t, "selfK", seq.SelfKCache[l], bat.SelfKCache[l])
						bitsEqual(t, "selfV", seq.SelfVCache[l], bat.SelfVCache[l])
					}
					for _, tok := range []int{7, 60, 3} {
						bitsEqual(t, "logits", dec.ForwardToken(tok, seq), dec.ForwardToken(tok, bat))
					}
				}
			}
		}
	}
}

func bitsEqual(t *testing.T, what string, want, got []float32) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s length %d vs %d", what, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
			t.Fatalf("%s[%d]: %v vs %v", what, i, got[i], want[i])
		}
	}
}
