package speaker

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

type speechBrainFeatureFixture struct {
	NumMels  int         `json:"num_mels"`
	Frames   int         `json:"frames"`
	Features [][]float32 `json:"features"`
}

// TestSpeechBrainECAPAReferenceFeatureParity is an opt-in local parity test. It
// expects a converted checkpoint at models/speaker-ecapa-voxceleb.safetensors
// and reference features/embedding produced by scripts/speechbrain_ecapa_reference.py:
//
//	python scripts/speechbrain_ecapa_reference.py \
//	  --source models/speechbrain-ecapa-voxceleb \
//	  --audio testdata/jfk.wav \
//	  --output /workspace/tmp/ecapa_jfk_reference.json \
//	  --features-output /workspace/tmp/ecapa_jfk_features.json
func TestSpeechBrainECAPAReferenceFeatureParity(t *testing.T) {
	if os.Getenv("SPEECHBRAIN_ECAPA_PARITY_TEST") != "1" {
		t.Skip("set SPEECHBRAIN_ECAPA_PARITY_TEST=1 after generating reference fixtures")
	}
	model, err := LoadSpeechBrainECAPASafetensors("../../models/speaker-ecapa-voxceleb.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	var ff speechBrainFeatureFixture
	readJSON(t, "/workspace/tmp/ecapa_jfk_features.json", &ff)
	flat := make([]float32, ff.NumMels*ff.Frames)
	for m := 0; m < ff.NumMels; m++ {
		copy(flat[m*ff.Frames:], ff.Features[m])
	}
	got := model.Embed(flat, ff.Frames)
	var ref struct {
		Embedding []float32 `json:"embedding"`
	}
	readJSON(t, "/workspace/tmp/ecapa_jfk_reference.json", &ref)
	if c := CosineSimilarity(got, ref.Embedding); c < 0.999 {
		t.Fatalf("Go/SpeechBrain embedding cosine=%f want >=0.999", c)
	}
	if math.Abs(float64(CosineSimilarity(got, got)-1)) > 1e-5 {
		t.Fatalf("self cosine drift")
	}
}

func readJSON(t *testing.T, path string, out any) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, out); err != nil {
		t.Fatal(err)
	}
}
