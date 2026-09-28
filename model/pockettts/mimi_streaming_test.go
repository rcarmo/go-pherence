package pockettts

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedMimiBatchMatchesFrameStreaming(t *testing.T) {
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
	latents := append(append([]float32(nil), ref.RawLatent...), ref.RawLatent...)
	batchState := model.NewStateForFrames(2)
	batch, err := model.Decode(latents, 2, batchState)
	if err != nil {
		t.Fatal(err)
	}
	streamState := model.NewStateForFrames(2)
	first, err := model.Decode(ref.RawLatent, 1, streamState)
	if err != nil {
		t.Fatal(err)
	}
	second, err := model.Decode(ref.RawLatent, 1, streamState)
	if err != nil {
		t.Fatal(err)
	}
	streamed := append(first, second...)
	if len(batch) != 2*SamplesPerFrame || len(streamed) != len(batch) {
		t.Fatalf("lengths batch=%d stream=%d", len(batch), len(streamed))
	}
	maxDiff := float64(0)
	for i := range batch {
		maxDiff = max(maxDiff, math.Abs(float64(batch[i]-streamed[i])))
	}
	if maxDiff > 3e-5 {
		t.Fatalf("batch/stream max diff=%g", maxDiff)
	}
}
