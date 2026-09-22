package pockettts

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type flowHeadReference struct {
	UpstreamCommit  string    `json:"upstream_commit"`
	WeightsRevision string    `json:"weights_revision"`
	WeightsSHA256   string    `json:"weights_sha256"`
	Times           []float32 `json:"times"`
	Output          []float32 `json:"output"`
}

func TestReleasedEnglishFlowHeadParity(t *testing.T) {
	path := releasedModel(t)
	file, err := safetensors.Open(path)
	if err != nil {
		t.Skipf("released Pocket TTS checkpoint unavailable: %v", err)
	}
	defer file.Close()
	model, err := LoadFlowHeadCPU(file, releasedConfig(t))
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("testdata/english-flow-head-reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref flowHeadReference
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	if ref.UpstreamCommit != UpstreamCommit || ref.WeightsRevision != EnglishWeightsRevision || len(ref.Output) != 32 || len(ref.Times) != 2 {
		t.Fatalf("fixture=%+v", ref)
	}
	condition := make([]float32, 1024)
	input := make([]float32, 32)
	for i := range condition {
		condition[i] = float32((i*17)%101-50) / 64
	}
	for i := range input {
		input[i] = float32((i*13)%31-15) / 32
	}
	got := make([]float32, 32)
	if err := model.Forward(got, condition, ref.Times, input); err != nil {
		t.Fatal(err)
	}
	maxDiff := float64(0)
	for i := range got {
		d := math.Abs(float64(got[i] - ref.Output[i]))
		maxDiff = max(maxDiff, d)
	}
	if maxDiff > 2e-4 {
		t.Fatalf("flow head max diff=%g\ngot=%v\nwant=%v", maxDiff, got, ref.Output)
	}
}
