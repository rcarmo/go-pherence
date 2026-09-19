package whisper

import "testing"

func TestDecoderForwardTokenShape(t *testing.T) {
	cfg := Tiny()
	dec := NewDecoder(cfg)
	dModel := cfg.DecoderDModel

	// Initialize weights with zeros
	dec.TokenEmbed = make([]float32, cfg.VocabSize*dModel)
	dec.PosEmbed = make([]float32, cfg.MaxDecoderLength*dModel)
	dec.FinalLNWeight = ones(dModel)
	dec.FinalLNBias = make([]float32, dModel)

	for i := range dec.Layers {
		l := &dec.Layers[i]
		l.SelfAttnLNWeight = ones(dModel)
		l.SelfAttnLNBias = make([]float32, dModel)
		l.SelfQWeight = make([]float32, dModel*dModel)
		l.SelfQBias = make([]float32, dModel)
		l.SelfKWeight = make([]float32, dModel*dModel)
		l.SelfKBias = make([]float32, dModel)
		l.SelfVWeight = make([]float32, dModel*dModel)
		l.SelfVBias = make([]float32, dModel)
		l.SelfOWeight = make([]float32, dModel*dModel)
		l.SelfOBias = make([]float32, dModel)

		l.CrossAttnLNWeight = ones(dModel)
		l.CrossAttnLNBias = make([]float32, dModel)
		l.CrossQWeight = make([]float32, dModel*dModel)
		l.CrossQBias = make([]float32, dModel)
		l.CrossKWeight = make([]float32, dModel*dModel)
		l.CrossKBias = make([]float32, dModel)
		l.CrossVWeight = make([]float32, dModel*dModel)
		l.CrossVBias = make([]float32, dModel)
		l.CrossOWeight = make([]float32, dModel*dModel)
		l.CrossOBias = make([]float32, dModel)

		l.MLPLNWeight = ones(dModel)
		l.MLPLNBias = make([]float32, dModel)
		l.FC1Weight = make([]float32, cfg.DecoderFFNDim*dModel)
		l.FC1Bias = make([]float32, cfg.DecoderFFNDim)
		l.FC2Weight = make([]float32, dModel*cfg.DecoderFFNDim)
		l.FC2Bias = make([]float32, dModel)
	}

	// Simulated encoder output: 240 frames × d_model
	encLen := 240
	encoderOutput := make([]float32, encLen*dModel)

	state := NewDecoderState(cfg, encoderOutput, encLen, dec)

	// Forward 3 tokens
	for i := 0; i < 3; i++ {
		logits := dec.ForwardToken(50258, state) // <|startoftranscript|>
		if len(logits) != cfg.VocabSize {
			t.Fatalf("token %d: logits length=%d want %d", i, len(logits), cfg.VocabSize)
		}
	}

	if state.Pos != 3 {
		t.Fatalf("state.Pos=%d want 3", state.Pos)
	}

	// Self-attention KV cache should have 3 entries per layer
	for l := 0; l < cfg.DecoderLayers; l++ {
		expectedKVLen := 3 * dModel
		if len(state.SelfKCache[l]) != expectedKVLen {
			t.Fatalf("layer %d KV cache len=%d want %d", l, len(state.SelfKCache[l]), expectedKVLen)
		}
	}
}

func TestCrossAttentionCached(t *testing.T) {
	cfg := Tiny()
	dec := NewDecoder(cfg)
	dModel := cfg.DecoderDModel

	// Set up minimal weights
	dec.TokenEmbed = make([]float32, cfg.VocabSize*dModel)
	dec.PosEmbed = make([]float32, cfg.MaxDecoderLength*dModel)
	dec.FinalLNWeight = ones(dModel)
	dec.FinalLNBias = make([]float32, dModel)

	for i := range dec.Layers {
		l := &dec.Layers[i]
		l.SelfAttnLNWeight = ones(dModel)
		l.SelfAttnLNBias = make([]float32, dModel)
		l.SelfQWeight = make([]float32, dModel*dModel)
		l.SelfQBias = make([]float32, dModel)
		l.SelfKWeight = make([]float32, dModel*dModel)
		l.SelfKBias = make([]float32, dModel)
		l.SelfVWeight = make([]float32, dModel*dModel)
		l.SelfVBias = make([]float32, dModel)
		l.SelfOWeight = make([]float32, dModel*dModel)
		l.SelfOBias = make([]float32, dModel)

		l.CrossAttnLNWeight = ones(dModel)
		l.CrossAttnLNBias = make([]float32, dModel)
		l.CrossQWeight = make([]float32, dModel*dModel)
		l.CrossQBias = make([]float32, dModel)
		l.CrossKWeight = make([]float32, dModel*dModel)
		l.CrossKBias = make([]float32, dModel)
		l.CrossVWeight = make([]float32, dModel*dModel)
		l.CrossVBias = make([]float32, dModel)
		l.CrossOWeight = make([]float32, dModel*dModel)
		l.CrossOBias = make([]float32, dModel)

		l.MLPLNWeight = ones(dModel)
		l.MLPLNBias = make([]float32, dModel)
		l.FC1Weight = make([]float32, cfg.DecoderFFNDim*dModel)
		l.FC1Bias = make([]float32, cfg.DecoderFFNDim)
		l.FC2Weight = make([]float32, dModel*cfg.DecoderFFNDim)
		l.FC2Bias = make([]float32, dModel)
	}

	encLen := 100
	encoderOutput := make([]float32, encLen*dModel)
	state := NewDecoderState(cfg, encoderOutput, encLen, dec)

	// Cross-K/V should be pre-computed with correct shape
	for l := 0; l < cfg.DecoderLayers; l++ {
		if len(state.CrossK[l]) != encLen*dModel {
			t.Fatalf("layer %d CrossK length=%d want %d", l, len(state.CrossK[l]), encLen*dModel)
		}
		if len(state.CrossV[l]) != encLen*dModel {
			t.Fatalf("layer %d CrossV length=%d want %d", l, len(state.CrossV[l]), encLen*dModel)
		}
	}
}
