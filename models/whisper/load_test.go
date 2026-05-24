package whisper

import (
	"math"
	"os"
	"testing"
)

// TestLoadWhisperTiny loads real whisper-tiny weights and verifies encoder output shape.
// Requires: models/whisper-tiny-hf/model.safetensors
func TestLoadWhisperTiny(t *testing.T) {
	modelPath := "../../models/whisper-tiny-hf/model.safetensors"
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("whisper-tiny model not available:", err)
	}

	cfg := Tiny()
	enc, dec, err := LoadModel(modelPath, cfg)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	// Verify encoder weights loaded
	if len(enc.Conv1Weight) != cfg.EncoderDModel*cfg.NumMelBins*3 {
		t.Fatalf("Conv1Weight length=%d want %d", len(enc.Conv1Weight), cfg.EncoderDModel*cfg.NumMelBins*3)
	}
	if len(enc.Conv1Bias) != cfg.EncoderDModel {
		t.Fatalf("Conv1Bias length=%d want %d", len(enc.Conv1Bias), cfg.EncoderDModel)
	}

	// Verify decoder weights loaded
	if len(dec.TokenEmbed) == 0 {
		t.Fatal("TokenEmbed is empty")
	}
	if len(dec.FinalLNWeight) != cfg.DecoderDModel {
		t.Fatalf("FinalLNWeight length=%d want %d", len(dec.FinalLNWeight), cfg.DecoderDModel)
	}

	// Verify layer weights for first encoder layer
	l0 := &enc.Layers[0]
	if len(l0.QWeight) == 0 {
		t.Fatal("encoder layer 0 QWeight is empty")
	}
	if len(l0.FC1Weight) == 0 {
		t.Fatal("encoder layer 0 FC1Weight is empty")
	}

	t.Logf("Loaded whisper-tiny: enc=%d layers, dec=%d layers, vocab=%d",
		len(enc.Layers), len(dec.Layers), len(dec.TokenEmbed)/cfg.DecoderDModel)
}

// TestWhisperTinyEncoderForward runs encoder forward with real weights on a sine wave.
func TestWhisperTinyEncoderForward(t *testing.T) {
	modelPath := "../../models/whisper-tiny-hf/model.safetensors"
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("whisper-tiny model not available:", err)
	}

	cfg := Tiny()
	enc, _, err := LoadModel(modelPath, cfg)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	// Generate 1 second of 440Hz tone
	samples := make([]float32, 16000)
	for i := range samples {
		samples[i] = float32(0.5 * math.Sin(2*math.Pi*440*float64(i)/16000))
	}

	// Compute mel
	T := 100 // ~1s worth of frames for a short test
	mel := make([]float32, cfg.NumMelBins*T)
	for i := range mel {
		mel[i] = float32(i%80) * 0.01 // Synthetic mel-like input
	}

	// Run encoder forward
	out := enc.Forward(mel, T)

	expectedT := (T+2*1-3)/2 + 1 // conv2 stride=2
	expectedLen := expectedT * cfg.EncoderDModel

	if len(out) != expectedLen {
		t.Fatalf("encoder output length=%d want %d (T'=%d)", len(out), expectedLen, expectedT)
	}

	// Verify output is non-trivial (not all zeros)
	var sum float64
	for _, v := range out {
		sum += math.Abs(float64(v))
	}
	if sum == 0 {
		t.Fatal("encoder output is all zeros with real weights")
	}
	t.Logf("encoder output: len=%d, mean_abs=%.6f", len(out), sum/float64(len(out)))
}
