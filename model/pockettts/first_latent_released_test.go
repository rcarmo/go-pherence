package pockettts

import (
	"encoding/json"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"math"
	"os"
	"reflect"
	"testing"
)

type firstLatentReference struct {
	VoiceRevision string    `json:"voice_revision"`
	VoiceSHA256   string    `json:"voice_sha256"`
	Text          string    `json:"text"`
	TokenIDs      []uint32  `json:"token_ids"`
	Temperature   float32   `json:"temperature"`
	DecodeSteps   int       `json:"decode_steps"`
	EOSThreshold  float32   `json:"eos_threshold"`
	BOSHidden     []float32 `json:"bos_hidden"`
	EOSLogit      float32   `json:"eos_logit"`
	Noise         []float32 `json:"noise"`
	Velocity      []float32 `json:"velocity"`
	Latent        []float32 `json:"latent"`
	EOS           bool      `json:"eos"`
	FinalOffset   int       `json:"final_offset"`
}

func TestReleasedAlbaFirstLatentParity(t *testing.T) {
	modelPath := releasedModel(t)
	voicePath := releasedVoice(t)
	file, err := safetensors.Open(modelPath)
	if err != nil {
		t.Skipf("released model unavailable: %v", err)
	}
	defer file.Close()
	cfg := releasedConfig(t)
	lm, err := LoadFlowLMCPU(file, cfg)
	if err != nil {
		t.Fatal(err)
	}
	flow, err := LoadFlowHeadCPU(file, cfg)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("testdata/english-alba-first-latent.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref firstLatentReference
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	tok := releasedTokenizer(t)
	ids, err := tok.Encode(ref.Text)
	if err != nil || !reflect.DeepEqual(ids, ref.TokenIDs) {
		t.Fatalf("tokens=%v want=%v err=%v", ids, ref.TokenIDs, err)
	}
	state, err := LoadVoiceState(voicePath, lm.Transformer, len(ids)+1)
	if err != nil {
		t.Fatal(err)
	}
	if err := lm.PromptText(state, ids); err != nil {
		t.Fatal(err)
	}
	noise := make([]float32, 32)
	std := float32(math.Sqrt(float64(ref.Temperature)))
	for i := range noise {
		noise[i] = float32((i*7)%19-9) / 16 * std
	}
	if !reflect.DeepEqual(noise, ref.Noise) {
		t.Fatalf("noise differs got=%v want=%v", noise, ref.Noise)
	}
	row := make([]float32, cfg.FlowLM.Transformer.DModel)
	if err := lm.Input.Forward(row, lm.BOS); err != nil {
		t.Fatal(err)
	}
	hidden, err := lm.Transformer.Step(row, state)
	if err != nil {
		t.Fatal(err)
	}
	hiddenDiff := float64(0)
	for i := range hidden {
		hiddenDiff = max(hiddenDiff, math.Abs(float64(hidden[i]-ref.BOSHidden[i])))
	}
	eosRow := make([]float32, 1)
	if err := lm.EOS.Forward(eosRow, hidden); err != nil {
		t.Fatal(err)
	}
	velocity := make([]float32, 32)
	if err := flow.Forward(velocity, hidden, []float32{0, 1}, noise); err != nil {
		t.Fatal(err)
	}
	velocityDiff := float64(0)
	for i := range velocity {
		velocityDiff = max(velocityDiff, math.Abs(float64(velocity[i]-ref.Velocity[i])))
	}
	got := make([]float32, 32)
	for i := range got {
		got[i] = noise[i] + velocity[i]
	}
	eos := eosRow[0] > ref.EOSThreshold
	maxDiff := float64(0)
	for i := range got {
		maxDiff = max(maxDiff, math.Abs(float64(got[i]-ref.Latent[i])))
	}
	if hiddenDiff > 3e-5 || math.Abs(float64(eosRow[0]-ref.EOSLogit)) > 3e-5 || velocityDiff > 5e-4 || maxDiff > 5e-4 || eos != ref.EOS || state.Position != ref.FinalOffset {
		t.Fatalf("first latent hidden=%g velocity=%g latent=%g eos=%g/%g bool=%v/%v offset=%d/%d", hiddenDiff, velocityDiff, maxDiff, eosRow[0], ref.EOSLogit, eos, ref.EOS, state.Position, ref.FinalOffset)
	}
}
