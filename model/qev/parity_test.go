package qev

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model"
)

type llamaCppQEVFixture struct {
	Schema string `json:"schema"`
	Oracle struct {
		Revision    string            `json:"revision"`
		ModelSHA256 string            `json:"model_sha256"`
		Environment map[string]string `json:"environment"`
	} `json:"oracle"`
	Inputs struct {
		SharedSHA256  string `json:"shared_sha256"`
		ContextSHA256 string `json:"context_sha256"`
		FieldSHA256   string `json:"field_sha256"`
		SharedText    string `json:"shared_text"`
		ContextText   string `json:"context_text"`
		FieldText     string `json:"field_text"`
	} `json:"inputs"`
	Tokens struct {
		Shared    []int `json:"shared"`
		Context   []int `json:"context"`
		TruePath  []int `json:"true_path"`
		FalsePath []int `json:"false_path"`
	} `json:"tokens"`
	Result struct {
		Mode           string    `json:"mode"`
		Rows           int       `json:"rows"`
		SharedTokens   int       `json:"shared_tokens"`
		ContextTokens  int       `json:"context_tokens"`
		DecisionRounds int       `json:"decision_rounds"`
		FieldIndex     int       `json:"field_index"`
		CandidateIndex int       `json:"candidate_index"`
		ScoredNodes    int       `json:"scored_nodes"`
		Tree           bool      `json:"tree"`
		Probabilities  []float64 `json:"probabilities"`
	} `json:"result"`
}

func loadLlamaCppQEVFixture(t *testing.T) llamaCppQEVFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/llamacpp-gemma4-12b-boolean.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture llamaCppQEVFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	return fixture
}

func TestPinnedLlamaCppFixtureContract(t *testing.T) {
	fixture := loadLlamaCppQEVFixture(t)
	if fixture.Schema != "go-pherence-qev-llamacpp-parity-v1" || fixture.Oracle.Revision != V1Provenance.LlamaRevision || fixture.Oracle.ModelSHA256 != V1Provenance.ModelSHA256 {
		t.Fatalf("oracle=%+v schema=%q", fixture.Oracle, fixture.Schema)
	}
	if fixture.Oracle.Environment["DECIDE_TREE"] != "1" || fixture.Oracle.Environment["DECIDE_NSEQ"] != "12" {
		t.Fatalf("oracle environment=%v", fixture.Oracle.Environment)
	}
	for name, item := range map[string]struct{ text, want string }{
		"shared": {fixture.Inputs.SharedText, fixture.Inputs.SharedSHA256}, "context": {fixture.Inputs.ContextText, fixture.Inputs.ContextSHA256}, "field": {fixture.Inputs.FieldText, fixture.Inputs.FieldSHA256},
	} {
		sum := sha256.Sum256([]byte(item.text))
		if got := hex.EncodeToString(sum[:]); got != item.want {
			t.Fatalf("%s text sha256=%s, want %s", name, got, item.want)
		}
	}
	if len(fixture.Tokens.Shared) != fixture.Result.SharedTokens || len(fixture.Tokens.Context) != fixture.Result.ContextTokens || fixture.Result.Rows != 4 || fixture.Result.DecisionRounds != 1 || fixture.Result.CandidateIndex != 0 || !fixture.Result.Tree || len(fixture.Result.Probabilities) != 2 {
		t.Fatalf("fixture contract=%+v", fixture)
	}
}

func TestGemma4NVIDIAReleasedModelMatchesPinnedLlamaCppDecision(t *testing.T) {
	modelPath, tokenizerDir := os.Getenv("GO_PHERENCE_GEMMA4_12B"), os.Getenv("GO_PHERENCE_GEMMA4_12B_TOKENIZER")
	if modelPath == "" || tokenizerDir == "" {
		t.Skip("set GO_PHERENCE_GEMMA4_12B and GO_PHERENCE_GEMMA4_12B_TOKENIZER")
	}
	fixture := loadLlamaCppQEVFixture(t)
	tok, err := tokenizer.LoadWithConfig(tokenizerDir)
	if err != nil {
		t.Fatal(err)
	}
	m, err := model.LoadGemma4GGUFAsLlama(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	m.Tok = tok
	gpu, err := model.NewGemma4NVIDIA(m)
	if err != nil {
		t.Fatal(err)
	}
	defer gpu.Close()
	prompt := append(append([]int(nil), fixture.Tokens.Shared...), fixture.Tokens.Context...)
	trueToken, falseToken := fixture.Tokens.TruePath[len(fixture.Tokens.TruePath)-2], fixture.Tokens.FalsePath[len(fixture.Tokens.FalsePath)-2]
	branches := []Branch{{Tokens: append([]int(nil), fixture.Tokens.TruePath[:len(fixture.Tokens.TruePath)-2]...), CandidateTokens: []int{trueToken, falseToken}}}
	got, err := (&Gemma4NVIDIAScorer{Model: m, GPU: gpu}).ScoreContext(context.Background(), prompt, branches, false)
	if err != nil {
		t.Fatal(err)
	}
	winner := 0
	if got[0][1] > got[0][0] {
		winner = 1
	}
	probability := constrainedProbability(got[0], winner)
	if winner != fixture.Result.CandidateIndex {
		t.Fatalf("winner=%d want %d logits=%v", winner, fixture.Result.CandidateIndex, got[0])
	}
	if math.Abs(probability-fixture.Result.Probabilities[winner]) > 1e-9 {
		t.Fatalf("probability=%.15g want %.15g", probability, fixture.Result.Probabilities[winner])
	}
}
