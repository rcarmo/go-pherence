package whisper

import "testing"

func TestChunkedTranscribeShort(t *testing.T) {
	cfg := Tiny()
	enc := NewEncoder(cfg)
	dec := NewDecoder(cfg)
	dModel := cfg.EncoderDModel

	// Minimal weight init
	enc.Conv1Weight = make([]float32, dModel*cfg.NumMelBins*3)
	enc.Conv1Bias = make([]float32, dModel)
	enc.Conv2Weight = make([]float32, dModel*dModel*3)
	enc.Conv2Bias = make([]float32, dModel)
	for i := range enc.Layers {
		l := &enc.Layers[i]
		l.AttnLNWeight = ones(dModel)
		l.AttnLNBias = make([]float32, dModel)
		l.QWeight = make([]float32, dModel*dModel)
		l.QBias = make([]float32, dModel)
		l.KWeight = make([]float32, dModel*dModel)
		l.KBias = make([]float32, dModel)
		l.VWeight = make([]float32, dModel*dModel)
		l.VBias = make([]float32, dModel)
		l.OWeight = make([]float32, dModel*dModel)
		l.OBias = make([]float32, dModel)
		l.MLPLNWeight = ones(dModel)
		l.MLPLNBias = make([]float32, dModel)
		l.FC1Weight = make([]float32, cfg.EncoderFFNDim*dModel)
		l.FC1Bias = make([]float32, cfg.EncoderFFNDim)
		l.FC2Weight = make([]float32, dModel*cfg.EncoderFFNDim)
		l.FC2Bias = make([]float32, dModel)
	}

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

	w := &Whisper{Encoder: enc, Decoder: dec, Config: cfg}

	// 5 seconds of audio (single chunk, no splitting needed)
	samples := make([]float32, 5*16000)
	segs, err := w.ChunkedTranscribe(samples, 1.0)
	if err != nil {
		t.Fatalf("ChunkedTranscribe: %v", err)
	}
	// With zero weights, decoder will quickly hit EOT
	t.Logf("segments: %d", len(segs))
}

func TestAppendNonOverlapping(t *testing.T) {
	existing := []Segment{
		{Start: 0, End: 2, Text: "hello"},
	}
	new := []Segment{
		{Start: 1.5, End: 3, Text: "overlap"}, // should be skipped
		{Start: 2.5, End: 4, Text: "world"},   // should be kept
	}
	result := appendNonOverlapping(existing, new)
	if len(result) != 2 {
		t.Fatalf("result length=%d want 2", len(result))
	}
	if result[1].Text != "world" {
		t.Fatalf("result[1].Text=%q want 'world'", result[1].Text)
	}
}
