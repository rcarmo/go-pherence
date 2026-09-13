package omnivoice

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
)

type codecEncoderFixture struct {
	Waveform       []float32 `json:"waveform"`
	SemanticFrames int       `json:"semantic_frames"`
	Semantic       []float32 `json:"semantic"`
	Codes          []int     `json:"codes"`
	Books          int       `json:"books"`
	Frames         int       `json:"frames"`
}

func TestRealCodecEncoderParity(t *testing.T) {
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
	script := filepath.Clean(filepath.Join(wd, "../../scripts/omnivoice-encoder-fixture.py"))
	if _, err = os.Stat(script); err != nil {
		t.Fatalf("missing fixture script: %v", err)
	}
	fixturePath := filepath.Join(t.TempDir(), "codec-encoder-fixture.json")
	cmd := exec.Command(python, script, "--model", modelPath, "--out", fixturePath)
	cmd.Env = os.Environ()
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("fixture generator: %v\n%s", err, out)
	}
	raw, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	var fixture codecEncoderFixture
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Books != 8 || fixture.Frames != fixture.SemanticFrames {
		t.Fatalf("bad fixture dimensions books=%d frames=%d semantic=%d", fixture.Books, fixture.Frames, fixture.SemanticFrames)
	}
	weights, err := loader.LoadCodecEncoder(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	encoder, err := NewCodecEncoder(weights)
	if err != nil {
		t.Fatal(err)
	}
	codes, err := encoder.EncodeFeatures(context.Background(), fixture.Waveform, fixture.Semantic, fixture.SemanticFrames)
	if err != nil {
		t.Fatal(err)
	}
	if len(codes) != len(fixture.Codes) {
		t.Fatalf("codes=%d want %d", len(codes), len(fixture.Codes))
	}
	for i, got := range codes {
		if got != fixture.Codes[i] {
			t.Fatalf("code %d got %d want %d", i, got, fixture.Codes[i])
		}
	}
	var allocErr error
	allocs := testing.AllocsPerRun(1, func() {
		_, allocErr = encoder.EncodeFeatures(context.Background(), fixture.Waveform, fixture.Semantic, fixture.SemanticFrames)
	})
	if allocErr != nil {
		t.Fatal(allocErr)
	}
	t.Logf("encoder parity ok for %d books x %d frames; EncodeFeatures allocations=%g", fixture.Books, fixture.Frames, allocs)
}
