package whisper

import (
	"math"
	"testing"
)

// TestEndToEndPipelineShape exercises the full pipeline from audio samples
// through to token output, verifying shape correctness at each stage.
// Uses zero weights (outputs will be zero/uniform) but validates the data flow.
func TestEndToEndPipelineShape(t *testing.T) {
	cfg := Tiny()
	enc, dec := buildZeroWeightModel(t, cfg)

	w := &Whisper{Encoder: enc, Decoder: dec, Config: cfg}

	// Generate 2 seconds of 440Hz tone
	samples := make([]float32, 2*16000)
	for i := range samples {
		samples[i] = float32(0.5 * math.Sin(2*math.Pi*440*float64(i)/16000))
	}

	text, err := w.TranscribeFromSamples(samples)
	if err != nil {
		t.Fatalf("TranscribeFromSamples: %v", err)
	}
	// With zero weights, we expect empty or placeholder text
	t.Logf("transcription (zero weights): %q", text)
}

// TestEndToEndTimestamps verifies timestamp decoding produces valid segments.
func TestEndToEndTimestamps(t *testing.T) {
	cfg := Tiny()
	enc, dec := buildZeroWeightModel(t, cfg)

	w := &Whisper{Encoder: enc, Decoder: dec, Config: cfg}

	// 1 second of audio
	samples := make([]float32, 16000)

	segs, err := w.ChunkedTranscribe(samples, 1.0)
	if err != nil {
		t.Fatalf("ChunkedTranscribe: %v", err)
	}
	// Validate segment timing invariants
	for i, seg := range segs {
		if seg.End < seg.Start {
			t.Fatalf("segment %d: end %.2f < start %.2f", i, seg.End, seg.Start)
		}
		if seg.Start < 0 {
			t.Fatalf("segment %d: negative start %.2f", i, seg.Start)
		}
	}
	t.Logf("timestamp segments: %d", len(segs))
}

// TestTimestampTokenConversion validates timestamp token math.
func TestTimestampTokenConversion(t *testing.T) {
	// Token 50364 = 0.00s
	if s := TimestampToSeconds(TokenTimestampBegin); s != 0 {
		t.Fatalf("timestamp begin = %f want 0", s)
	}
	// Token 50364 + 50 = 1.00s (50 * 0.02)
	if s := TimestampToSeconds(TokenTimestampBegin + 50); math.Abs(s-1.0) > 0.001 {
		t.Fatalf("1s timestamp = %f want 1.0", s)
	}
	// Token 50364 + 1500 = 30.00s
	if s := TimestampToSeconds(TokenTimestampBegin + 1500); math.Abs(s-30.0) > 0.001 {
		t.Fatalf("30s timestamp = %f want 30.0", s)
	}
	// Non-timestamp token
	if IsTimestamp(100) {
		t.Fatal("100 should not be a timestamp")
	}
	if !IsTimestamp(TokenTimestampBegin) {
		t.Fatal("TokenTimestampBegin should be a timestamp")
	}
}

// TestSpecialTokenConstants validates special token ID assignments.
func TestSpecialTokenConstants(t *testing.T) {
	if TokenSOT <= 0 {
		t.Fatal("invalid SOT")
	}
	if TokenEOT <= 0 {
		t.Fatal("invalid EOT")
	}
	if TokenTranscribe <= TokenSOT {
		t.Fatal("transcribe should be after SOT")
	}
	if TokenTimestampBegin <= TokenNoTimestamps {
		t.Fatal("timestamp begin should be after notimestamps")
	}
}

func buildZeroWeightModel(t *testing.T, cfg Config) (*Encoder, *Decoder) {
	t.Helper()
	dModel := cfg.EncoderDModel

	enc := NewEncoder(cfg)
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

	dec := NewDecoder(cfg)
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

	return enc, dec
}
