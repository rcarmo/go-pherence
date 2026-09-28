package pockettts

import (
	"encoding/json"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"math"
	"os"
	"testing"
)

type mimiReference struct {
	RawLatent       []float32 `json:"raw_latent"`
	Waveform        []float32 `json:"waveform"`
	WaveformSamples int       `json:"waveform_samples"`
}

func TestReleasedMimiOneFrameParity(t *testing.T) {
	path := releasedModel(t)
	file, err := safetensors.Open(path)
	if err != nil {
		t.Skipf("released model unavailable: %v", err)
	}
	defer file.Close()
	model, err := LoadMimiDecoderCPU(file, releasedConfig(t))
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("testdata/english-alba-first-latent.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref mimiReference
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	got, err := model.Decode(ref.RawLatent, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != SamplesPerFrame || ref.WaveformSamples != SamplesPerFrame {
		t.Fatalf("samples=%d/%d", len(got), ref.WaveformSamples)
	}
	maxDiff := float64(0)
	for i := range got {
		maxDiff = max(maxDiff, math.Abs(float64(got[i]-ref.Waveform[i])))
	}
	if maxDiff > 3e-5 {
		t.Fatalf("Mimi max diff=%g first=%v want=%v", maxDiff, got[:8], ref.Waveform[:8])
	}
}
