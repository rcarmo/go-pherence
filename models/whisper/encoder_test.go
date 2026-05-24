package whisper

import "testing"

func TestEncoderForwardShape(t *testing.T) {
	cfg := Tiny()
	enc := NewEncoder(cfg)

	// Initialize conv weights with zeros (just test shape flow)
	enc.Conv1Weight = make([]float32, cfg.EncoderDModel*cfg.NumMelBins*3)
	enc.Conv1Bias = make([]float32, cfg.EncoderDModel)
	enc.Conv2Weight = make([]float32, cfg.EncoderDModel*cfg.EncoderDModel*3)
	enc.Conv2Bias = make([]float32, cfg.EncoderDModel)

	// Initialize layer weights
	for i := range enc.Layers {
		l := &enc.Layers[i]
		l.AttnLNWeight = ones(cfg.EncoderDModel)
		l.AttnLNBias = make([]float32, cfg.EncoderDModel)
		l.QWeight = make([]float32, cfg.EncoderDModel*cfg.EncoderDModel)
		l.QBias = make([]float32, cfg.EncoderDModel)
		l.KWeight = make([]float32, cfg.EncoderDModel*cfg.EncoderDModel)
		l.KBias = make([]float32, cfg.EncoderDModel)
		l.VWeight = make([]float32, cfg.EncoderDModel*cfg.EncoderDModel)
		l.VBias = make([]float32, cfg.EncoderDModel)
		l.OWeight = make([]float32, cfg.EncoderDModel*cfg.EncoderDModel)
		l.OBias = make([]float32, cfg.EncoderDModel)
		l.MLPLNWeight = ones(cfg.EncoderDModel)
		l.MLPLNBias = make([]float32, cfg.EncoderDModel)
		l.FC1Weight = make([]float32, cfg.EncoderFFNDim*cfg.EncoderDModel)
		l.FC1Bias = make([]float32, cfg.EncoderFFNDim)
		l.FC2Weight = make([]float32, cfg.EncoderDModel*cfg.EncoderFFNDim)
		l.FC2Bias = make([]float32, cfg.EncoderDModel)
	}

	// Simulate 3 seconds of audio (480 mel frames at 10ms hop)
	T := 480
	mel := make([]float32, cfg.NumMelBins*T)

	out := enc.Forward(mel, T)

	// After conv2 with stride=2: T' = (480 + 2 - 3)/2 + 1 = 240
	expectedT := (T+2*1-3)/2 + 1
	expectedLen := expectedT * cfg.EncoderDModel

	if len(out) != expectedLen {
		t.Fatalf("encoder output length=%d, want %d (T'=%d, d_model=%d)", len(out), expectedLen, expectedT, cfg.EncoderDModel)
	}
}

func TestSinusoidalPositionEmbedding(t *testing.T) {
	pe := sinusoidalPositionEmbedding(100, 64)
	if len(pe) != 100*64 {
		t.Fatalf("PE length=%d want %d", len(pe), 100*64)
	}
	// Position 0, dim 0 should be sin(0) = 0
	if pe[0] != 0 {
		t.Fatalf("PE[0,0]=%f want 0", pe[0])
	}
	// Position 0, dim 1 should be cos(0) = 1
	if pe[1] != 1 {
		t.Fatalf("PE[0,1]=%f want 1", pe[1])
	}
}

func TestGELU(t *testing.T) {
	x := []float32{0, 1, -1, 2}
	gelu(x)
	// GELU(0) ≈ 0
	if x[0] < -0.01 || x[0] > 0.01 {
		t.Fatalf("GELU(0)=%f want ~0", x[0])
	}
	// GELU(1) ≈ 0.841
	if x[1] < 0.8 || x[1] > 0.9 {
		t.Fatalf("GELU(1)=%f want ~0.841", x[1])
	}
	// GELU(-1) ≈ -0.159
	if x[2] < -0.2 || x[2] > -0.1 {
		t.Fatalf("GELU(-1)=%f want ~-0.159", x[2])
	}
}

func ones(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = 1
	}
	return s
}
