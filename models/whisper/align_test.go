package whisper

import (
	"os"
	"testing"
)

func TestForceAlign(t *testing.T) {
	modelPath := "../../models/whisper-tiny-hf/model.safetensors"
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("whisper-tiny model not available")
	}

	cfg := Tiny()
	enc, dec, err := LoadModel(modelPath, cfg)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	// Synthetic mel input
	T := 100
	melFlat := make([]float32, cfg.NumMelBins*T)
	for i := range melFlat {
		melFlat[i] = float32(i%80) * 0.01
	}

	encoderOutput := enc.Forward(melFlat, T)
	encLen := len(encoderOutput) / cfg.EncoderDModel
	state := NewDecoderState(cfg, encoderOutput, encLen, dec)

	// Force-align 5 synthetic tokens
	tokens := []int{100, 200, 300, 400, 500}
	audioDuration := 3.0

	aligns := ForceAlign(dec, state, tokens, cfg, audioDuration)
	if len(aligns) != len(tokens) {
		t.Fatalf("alignments length=%d want %d", len(aligns), len(tokens))
	}

	// Verify monotonicity
	for i := 1; i < len(aligns); i++ {
		if aligns[i].Start < aligns[i-1].Start {
			t.Fatalf("non-monotonic: align[%d].Start=%f < align[%d].Start=%f",
				i, aligns[i].Start, i-1, aligns[i-1].Start)
		}
	}

	// Verify timing within audio duration
	for i, a := range aligns {
		if a.Start < 0 || a.End > audioDuration+0.1 {
			t.Fatalf("align[%d] out of range: [%f, %f] for duration %f", i, a.Start, a.End, audioDuration)
		}
		if a.Token != tokens[i] {
			t.Fatalf("align[%d].Token=%d want %d", i, a.Token, tokens[i])
		}
	}

	t.Logf("Force alignment results:")
	for i, a := range aligns {
		t.Logf("  token %d: %.3fs - %.3fs", a.Token, a.Start, a.End)
		_ = i
	}
}

func TestSmoothAlignments(t *testing.T) {
	aligns := []WordAlignment{
		{Start: 1.0, End: 0},
		{Start: 0.5, End: 0}, // non-monotonic
		{Start: 2.0, End: 0},
	}
	smoothAlignments(aligns, 3.0)

	// Should be monotonic now
	if aligns[1].Start < aligns[0].Start {
		t.Fatalf("not monotonic after smoothing: %f < %f", aligns[1].Start, aligns[0].Start)
	}
	// Last end should be total duration
	if aligns[2].End != 3.0 {
		t.Fatalf("last end=%f want 3.0", aligns[2].End)
	}
}
