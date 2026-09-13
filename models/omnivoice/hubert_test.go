package omnivoice

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
)

type hubertFixture struct {
	Input    []float32 `json:"input"`
	Frames   int       `json:"frames"`
	Semantic []float32 `json:"semantic"`
}

func TestRealHubertParity(t *testing.T) {
	modelPath := os.Getenv("GO_PHERENCE_REAL_CODEC")
	if modelPath == "" {
		modelPath = "/workspace/projects/spock-tts/models/omnivoice/audio_tokenizer"
	}
	if _, err := os.Stat(filepath.Join(modelPath, "model.safetensors")); err != nil {
		t.Skip("set GO_PHERENCE_REAL_CODEC to a HiggsAudioV2 tokenizer directory")
	}
	python := os.Getenv("GO_PHERENCE_REFERENCE_PYTHON")
	if python == "" {
		t.Skip("set GO_PHERENCE_REFERENCE_PYTHON")
	}
	if _, err := os.Stat(python); err != nil {
		t.Skip("python fixture generator unavailable")
	}
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	script := filepath.Clean(filepath.Join(wd, "../../scripts/omnivoice-hubert-fixture.py"))
	if _, err = os.Stat(script); err != nil {
		t.Fatalf("missing fixture script: %v", err)
	}
	fixturePath := filepath.Join(t.TempDir(), "hubert-fixture.json")
	cmd := exec.Command(python, script, "--model", modelPath, "--out", fixturePath)
	cmd.Env = os.Environ()
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("fixture generator: %v\n%s", err, out)
	}
	raw, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	var fixture hubertFixture
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	weights, err := loader.LoadHubert(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	hubert, err := NewHubert(weights)
	if err != nil {
		t.Fatal(err)
	}
	got, frames, err := hubert.Extract(context.Background(), fixture.Input)
	if err != nil {
		t.Fatal(err)
	}
	if frames != fixture.Frames {
		t.Fatalf("frames=%d want %d", frames, fixture.Frames)
	}
	if len(got) != len(fixture.Semantic) {
		t.Fatalf("semantic=%d want %d", len(got), len(fixture.Semantic))
	}
	maxErr, rms := 0.0, 0.0
	for i, want := range fixture.Semantic {
		err := math.Abs(float64(got[i] - want))
		if err > maxErr {
			maxErr = err
		}
		rms += err * err
	}
	if len(got) > 0 {
		rms = math.Sqrt(rms / float64(len(got)))
	}
	if maxErr > 1e-3 {
		t.Fatalf("max semantic error=%g rms=%g", maxErr, rms)
	}
	var allocErr error
	allocs := testing.AllocsPerRun(1, func() {
		_, _, allocErr = hubert.Extract(context.Background(), fixture.Input)
	})
	if allocErr != nil {
		t.Fatal(allocErr)
	}
	t.Logf("hubert parity ok frames=%d max_err=%g rms=%g; Extract allocations=%g", frames, maxErr, rms, allocs)
}
