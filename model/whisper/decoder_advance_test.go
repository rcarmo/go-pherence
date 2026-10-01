package whisper

import (
	"math"
	"testing"
)

// AdvanceToken must leave exactly the state ForwardToken leaves, so all later
// logits are bit-identical (with and without original decoder compatibility).
func TestDecoderAdvanceTokenExact(t *testing.T) {
	for _, compat := range []bool{false, true} {
		cfg, dec, enc, frames := syntheticDecoderStateFixture()
		d, f := cfg.DecoderDModel, cfg.DecoderFFNDim
		seed := 0
		fill := func(n int, scale float32) []float32 {
			v := make([]float32, n)
			for i := range v {
				seed = (seed*1103515245 + 12345) & 0x7fffffff
				v[i] = (float32(seed%2001)/1000 - 1) * scale
			}
			return v
		}
		dec.TokenEmbed, dec.PosEmbed = fill(cfg.VocabSize*d, 0.5), fill(cfg.MaxDecoderLength*d, 0.3)
		dec.FinalLNWeight, dec.FinalLNBias = fill(d, 1), fill(d, 0.1)
		for l := range dec.Layers {
			L := &dec.Layers[l]
			L.SelfAttnLNWeight, L.SelfAttnLNBias, L.CrossAttnLNWeight, L.CrossAttnLNBias, L.MLPLNWeight, L.MLPLNBias = fill(d, 1), fill(d, 0.1), fill(d, 1), fill(d, 0.1), fill(d, 1), fill(d, 0.1)
			L.SelfQWeight, L.SelfKWeight, L.SelfVWeight, L.SelfOWeight, L.CrossQWeight, L.CrossOWeight = fill(d*d, 0.4), fill(d*d, 0.4), fill(d*d, 0.4), fill(d*d, 0.4), fill(d*d, 0.4), fill(d*d, 0.4)
			L.SelfQBias, L.SelfKBias, L.SelfVBias, L.SelfOBias, L.CrossQBias, L.CrossOBias = fill(d, 0.1), fill(d, 0.1), fill(d, 0.1), fill(d, 0.1), fill(d, 0.1), fill(d, 0.1)
			L.FC1Weight, L.FC1Bias, L.FC2Weight, L.FC2Bias = fill(f*d, 0.4), fill(f, 0.1), fill(d*f, 0.4), fill(d, 0.1)
		}
		a := NewDecoderState(cfg, enc, frames, dec)
		b := NewDecoderState(cfg, enc, frames, dec)
		if compat {
			a.applyOriginalDecoderCompatibility(frames)
			b.applyOriginalDecoderCompatibility(frames)
		}
		prompt := []int{1, 3, 0, 2, 3}
		for _, tok := range prompt {
			dec.ForwardToken(tok, a)
			dec.AdvanceToken(tok, b)
		}
		if a.Pos != b.Pos || a.LastToken != b.LastToken {
			t.Fatal("position", a.Pos, b.Pos, a.LastToken, b.LastToken)
		}
		for _, tok := range []int{2, 1} {
			la, lb := dec.ForwardToken(tok, a), dec.ForwardToken(tok, b)
			if len(la) != len(lb) || len(la) == 0 || la[0] == la[1] || math.IsNaN(float64(la[0])) {
				t.Fatal("logit shape or degenerate fixture", la)
			}
			for i := range la {
				if math.Float32bits(la[i]) != math.Float32bits(lb[i]) {
					t.Fatal("logits differ", compat, tok, i, la[i], lb[i])
				}
			}
		}
	}
}
