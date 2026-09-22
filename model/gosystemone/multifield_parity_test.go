package gosystemone

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

type llamaCppMultiFieldFixture struct {
	Schema       string          `json:"schema"`
	SchemaJSON   json.RawMessage `json:"schema_json"`
	Instructions string          `json:"instructions"`
	SharedText   string          `json:"shared_text"`
	SharedSHA256 string          `json:"shared_sha256"`
	SharedTokens []int           `json:"shared_tokens"`
	Contexts     []struct {
		Text     string `json:"text"`
		Rendered string `json:"rendered"`
		SHA256   string `json:"sha256"`
		Tokens   []int  `json:"tokens"`
	} `json:"contexts"`
	Oracle struct {
		Revision    string            `json:"revision"`
		ModelSHA256 string            `json:"model_sha256"`
		Environment map[string]string `json:"environment"`
		Results     []struct {
			Rows           int `json:"rows"`
			SharedTokens   int `json:"shared_tokens"`
			ContextTokens  int `json:"context_tokens"`
			DecisionRounds int `json:"decision_rounds"`
			Fields         []struct {
				FieldIndex     int       `json:"field_index"`
				CandidateIndex int       `json:"candidate_index"`
				ScoredNodes    int       `json:"scored_nodes"`
				Tree           int       `json:"tree"`
				Probabilities  []float64 `json:"probs"`
			} `json:"fields"`
		} `json:"results"`
	} `json:"oracle"`
}

func loadLlamaCppMultiFieldFixture(t *testing.T) llamaCppMultiFieldFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/llamacpp-go-system-one-gemma4-12b-multifield.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture llamaCppMultiFieldFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	return fixture
}

func TestPinnedLlamaCppMultiFieldFixtureContract(t *testing.T) {
	fixture := loadLlamaCppMultiFieldFixture(t)
	if fixture.Schema != "go-pherence-go-system-one-llamacpp-parity-v2" || fixture.Oracle.Revision != V1Provenance.LlamaRevision || fixture.Oracle.ModelSHA256 != V1Provenance.ModelSHA256 {
		t.Fatalf("fixture pins schema=%q revision=%q model=%q", fixture.Schema, fixture.Oracle.Revision, fixture.Oracle.ModelSHA256)
	}
	if fixture.Oracle.Environment["DECIDE_TREE"] != "1" || fixture.Oracle.Environment["DECIDE_NSEQ"] != "12" {
		t.Fatalf("oracle environment=%v", fixture.Oracle.Environment)
	}
	if len(fixture.Contexts) != 2 || len(fixture.Oracle.Results) != 2 {
		t.Fatalf("contexts=%d results=%d", len(fixture.Contexts), len(fixture.Oracle.Results))
	}
	assertTextHash := func(name, text, want string) {
		t.Helper()
		sum := sha256.Sum256([]byte(text))
		if got := hex.EncodeToString(sum[:]); got != want {
			t.Fatalf("%s sha256=%s want=%s", name, got, want)
		}
	}
	assertTextHash("shared", fixture.SharedText, fixture.SharedSHA256)
	for i := range fixture.Contexts {
		assertTextHash("context", fixture.Contexts[i].Rendered, fixture.Contexts[i].SHA256)
		result := fixture.Oracle.Results[i]
		if result.SharedTokens != len(fixture.SharedTokens) || result.ContextTokens != len(fixture.Contexts[i].Tokens) || result.Rows != 9 || result.DecisionRounds != 1 || len(result.Fields) != 2 {
			t.Fatalf("result[%d]=%+v", i, result)
		}
	}
}

func TestGoSystemOneNVIDIAMultiFieldReleasedModelMatchesPinnedLlamaCpp(t *testing.T) {
	modelPath, tokenizerDir := os.Getenv("GO_PHERENCE_GO_SYSTEM_ONE_GEMMA4_12B"), os.Getenv("GO_PHERENCE_GO_SYSTEM_ONE_GEMMA4_12B_TOKENIZER")
	if modelPath == "" || tokenizerDir == "" {
		t.Skip("set GO_PHERENCE_GO_SYSTEM_ONE_GEMMA4_12B and GO_PHERENCE_GO_SYSTEM_ONE_GEMMA4_12B_TOKENIZER")
	}
	fixture := loadLlamaCppMultiFieldFixture(t)
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
	scorer := &Gemma4NVIDIAScorer{Model: m, GPU: gpu}
	defer scorer.Close()
	contexts := make([]string, len(fixture.Contexts))
	for i := range fixture.Contexts {
		contexts[i] = fixture.Contexts[i].Text
	}
	response, err := (&Engine{Tokenizer: tok, Scorer: scorer, BOSToken: m.Config.BOSTokenID}).Decide(context.Background(), Request{
		Model: fixture.Schema, Instructions: fixture.Instructions, Schema: fixture.SchemaJSON, Contexts: contexts, Mode: ModeTree,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Results) != len(fixture.Oracle.Results) {
		t.Fatalf("results=%d want=%d", len(response.Results), len(fixture.Oracle.Results))
	}
	compiled, err := CompileSchema(fixture.SchemaJSON, fixture.Instructions)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range fixture.Oracle.Results {
		for _, oracleField := range want.Fields {
			field := compiled.Fields[oracleField.FieldIndex]
			candidate := field.Candidates[oracleField.CandidateIndex]
			gotField := response.Results[i].Fields[field.Name]
			if string(gotField.Value) != string(candidate.Value) {
				t.Fatalf("context=%d field=%q value=%s want=%s", i, field.Name, gotField.Value, candidate.Value)
			}
			wantProbability := oracleField.Probabilities[oracleField.CandidateIndex]
			if math.Abs(gotField.Probability-wantProbability) > 1e-6 {
				t.Fatalf("context=%d field=%q probability=%.15g want=%.15g", i, field.Name, gotField.Probability, wantProbability)
			}
			if !gotField.Tree || gotField.ScoredNodes != oracleField.ScoredNodes {
				t.Fatalf("context=%d field=%q tree=%v nodes=%d", i, field.Name, gotField.Tree, gotField.ScoredNodes)
			}
		}
	}
}
