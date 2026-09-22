package pockettts

import (
	"encoding/json"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"math"
	"os"
	"testing"
)

type flowLMReference struct {
	UpstreamCommit  string    `json:"upstream_commit"`
	WeightsRevision string    `json:"weights_revision"`
	TokenIDs        []uint32  `json:"token_ids"`
	LastHidden      []float32 `json:"last_hidden"`
	EOSLogit        float32   `json:"eos_logit"`
}

func TestReleasedEnglishFlowLMPrefillParity(t *testing.T) {
	path := releasedModel(t)
	file, err := safetensors.Open(path)
	if err != nil {
		t.Skipf("released Pocket TTS checkpoint unavailable: %v", err)
	}
	defer file.Close()
	model, err := LoadFlowLMCPU(file, releasedConfig(t))
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("testdata/english-flowlm-reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref flowLMReference
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	latent := make([]float32, 32)
	for i := range latent {
		latent[i] = float32((i*13)%31-15) / 32
	}
	hidden, eos, err := model.Prefill(ref.TokenIDs, latent)
	if err != nil {
		t.Fatal(err)
	}
	streamed, streamEOS, state, err := model.PrefillStreaming(ref.TokenIDs, latent)
	if err != nil {
		t.Fatal(err)
	}
	maxDiff, streamDiff := float64(0), float64(0)
	for i := range hidden {
		maxDiff = max(maxDiff, math.Abs(float64(hidden[i]-ref.LastHidden[i])))
		streamDiff = max(streamDiff, math.Abs(float64(streamed[i]-hidden[i])))
	}
	if maxDiff > 3e-5 || math.Abs(float64(eos-ref.EOSLogit)) > 3e-5 || streamDiff > 3e-5 || math.Abs(float64(streamEOS-eos)) > 3e-5 || state.Position != len(ref.TokenIDs)+1 {
		t.Fatalf("FlowLM max diff=%g stream=%g eos=%g/%g stream_eos=%g position=%d", maxDiff, streamDiff, eos, ref.EOSLogit, streamEOS, state.Position)
	}
	if testing.Verbose() {
		t.Logf("FlowLM max_abs_diff=%g eos_abs_diff=%g", maxDiff, math.Abs(float64(eos-ref.EOSLogit)))
	}
}
