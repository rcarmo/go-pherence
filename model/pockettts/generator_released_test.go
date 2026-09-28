package pockettts

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

type generatorReference struct {
	Text     string    `json:"text"`
	TokenIDs []uint32  `json:"token_ids"`
	Noise    []float32 `json:"noise"`
	Waveform []float32 `json:"waveform"`
}

func TestReleasedEndToEndOneFrame(t *testing.T) {
	modelPath := releasedModel(t)
	voicePath := releasedVoice(t)
	gen, err := LoadGeneratorCPU(modelPath, releasedConfig(t))
	if err != nil {
		t.Skipf("released model unavailable: %v", err)
	}
	data, err := os.ReadFile("testdata/english-alba-first-latent.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref generatorReference
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	state, err := LoadVoiceState(voicePath, gen.FlowLM.Transformer, len(ref.TokenIDs)+1)
	if err != nil {
		t.Fatal(err)
	}
	noise := func(_ int, dst []float32) error { copy(dst, ref.Noise); return nil }
	got, err := gen.Generate(state, ref.TokenIDs, 1, 3, 1, 0, noise)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != SamplesPerFrame {
		t.Fatalf("samples=%d", len(got))
	}
	maxDiff := float64(0)
	for i := range got {
		maxDiff = max(maxDiff, math.Abs(float64(got[i]-ref.Waveform[i])))
	}
	if maxDiff > 3e-5 {
		t.Fatalf("end-to-end max diff=%g", maxDiff)
	}
}
