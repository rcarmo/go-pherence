package gosystemone

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model"
)

// This opt-in gate exercises the new prompt through the released tokenizer and
// GPU. It tests typed output invariants; it is not a labelled accuracy test.
func TestSystemOneReleasedModelTypes(t *testing.T) {
	path, dir := os.Getenv("GO_PHERENCE_GO_SYSTEM_ONE_GEMMA4_12B"), os.Getenv("GO_PHERENCE_GO_SYSTEM_ONE_GEMMA4_12B_TOKENIZER")
	if path == "" || dir == "" {
		t.Skip("set GO_PHERENCE_GO_SYSTEM_ONE_GEMMA4_12B and GO_PHERENCE_GO_SYSTEM_ONE_GEMMA4_12B_TOKENIZER")
	}
	tok, err := tokenizer.LoadWithConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	m, err := model.LoadGemma4GGUFAsLlama(path)
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
	engine := &Engine{Tokenizer: tok, Scorer: scorer, BOSToken: m.Config.BOSTokenID}
	request := SystemOneRequest{Model: "fixture", State: json.RawMessage(`{"ticket":"All production services are unavailable. Customers cannot connect."}`), Questions: json.RawMessage(`{
		"urgent":{"type":"noul","instructions":"Does this need urgent handling?"},
		"route":{"type":"choice","instructions":"Choose the team","criteria":{"technical operations":"Production outages","billing support":"Invoice questions"}},
		"severity":{"type":"score","instructions":"Rate incident severity","criteria":["Routine request","Degraded service","Total outage"]}
	}`)}
	for _, budget := range []int{0} {
		response, err := engine.SystemOne(context.Background(), request)
		if err != nil {
			t.Fatal(err)
		}
		noul := response.Answers["urgent"].(NoulAnswer)
		choice := response.Answers["route"].(ChoiceAnswer)
		score := response.Answers["severity"].(ScoreAnswer)
		if math.IsNaN(noul.Noul) || noul.Noul < 0 || noul.Noul > 1 || score.Score < 0 || score.Score > 2 {
			t.Fatal("typed answer out of range")
		}
		if len(choice.Probabilities) != 2 || len(score.Probabilities) != 3 || len(score.Legend) != 3 {
			t.Fatal("missing distribution or legend")
		}
		if math.Abs(score.Score-(score.Probabilities["1"]+2*score.Probabilities["2"])) > 1e-12 {
			t.Fatal("score is not the expected level")
		}
		for _, distribution := range []map[string]float64{choice.Probabilities, score.Probabilities} {
			sum := 0.0
			for _, p := range distribution {
				sum += p
			}
			if math.Abs(sum-1) > 1e-9 {
				t.Fatal("distribution is not normalised")
			}
		}
		data, _ := json.Marshal(response)
		t.Logf("rows=%d response=%s", budget, data)
	}
}
