package simplejev

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// This fixture compares model-free scoring only. Request, prompt, tokenizer,
// model and HTTP contracts are intentionally outside its scope.
func TestPinnedUpstreamScoringFixture(t *testing.T) {
	const scriptSHA = "6ca6ee503884b055a8d980434238332cf1859360947aa815ade4d545915d1ff3"
	const fixtureSHA = "a81f5d12818806af0c9c94dc95b9626448f74eeda836bf20d9dfe6d4e715fe46"
	fixturePath := filepath.Join("testdata", "upstream_scoring_v1.json")
	if err := verifyFixtureSHA(filepath.Join("..", "..", "scripts", "simplejev_oracle_scoring.py"), scriptSHA); err != nil {
		t.Fatal(err)
	}
	if err := verifyFixtureSHA(fixturePath, fixtureSHA); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema         int    `json:"schema"`
		OracleRepo     string `json:"oracle_repo"`
		OracleRevision string `json:"oracle_revision"`
		ScoringSHA     string `json:"scoring_sha256"`
		Branches       []struct {
			Branch   string   `json:"branch"`
			Question string   `json:"question"`
			Labels   []string `json:"labels"`
		} `json:"branches"`
		Cases []struct {
			Name   string      `json:"name"`
			Logits [][]float32 `json:"logits"`
			Choice struct {
				ID            string    `json:"id"`
				Probabilities []float64 `json:"probabilities"`
			} `json:"choice"`
			Score struct {
				Value         float64   `json:"value"`
				Probabilities []float64 `json:"probabilities"`
			} `json:"score"`
			Noul struct {
				Value float64 `json:"value"`
			} `json:"noul"`
		} `json:"cases"`
		Rejected map[string]string `json:"rejected"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != 1 || fixture.OracleRepo != "featherless-ai/simple-jev" || fixture.OracleRevision != "b02aa81c915a8193759b3cd33fef74721d6e005b" || fixture.ScoringSHA != "b1b1303ace5218fb40904e7aaf5a1ad01aa3d072c30678d45e921bb0e8613dfa" || len(fixture.Cases) != 3 ||
		len(fixture.Branches) != 3 || fixture.Branches[0].Branch != "0" || fixture.Branches[0].Question != "pick" || !reflect.DeepEqual(fixture.Branches[0].Labels, []string{"A", "B", "C"}) ||
		fixture.Branches[1].Branch != "1" || fixture.Branches[1].Question != "rating" || !reflect.DeepEqual(fixture.Branches[1].Labels, []string{"0", "1", "2", "3"}) ||
		fixture.Branches[2].Branch != "2" || fixture.Branches[2].Question != "binary" || !reflect.DeepEqual(fixture.Branches[2].Labels, []string{"1", "2", "3", "4", "5", "6", "7", "8", "9"}) ||
		!reflect.DeepEqual(fixture.Rejected, map[string]string{"missing_label": "ValueError", "extra_branch": "ValueError", "nonfinite": "ValueError"}) {
		t.Fatal("unexpected upstream fixture provenance or shape")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			if len(tc.Logits) != 3 || len(tc.Logits[0]) != 3 || len(tc.Logits[1]) != 4 || len(tc.Logits[2]) != 9 {
				t.Fatal("invalid oracle row lengths")
			}
			choice := make([]Label, 3)
			for i, name := range []string{"alpha", "beta", "gamma"} {
				choice[i] = Label{TokenID: i, ID: name, Logit: tc.Logits[0][i]}
			}
			winner, gotChoice, err := Choice(choice)
			if err != nil {
				t.Fatal(err)
			}
			if winner != tc.Choice.ID {
				t.Fatalf("choice=%s want=%s", winner, tc.Choice.ID)
			}
			checkOracleProbs(t, gotChoice, tc.Choice.Probabilities)

			score := make([]Label, 4)
			for i := range score {
				score[i] = Label{TokenID: i, ID: fixture.Branches[1].Labels[i], Logit: tc.Logits[1][i], Value: float64(i)}
			}
			gotScore, gotScoreProbs, err := Ordinal(score)
			if err != nil {
				t.Fatal(err)
			}
			checkOracleProbs(t, gotScoreProbs, tc.Score.Probabilities)
			if math.Abs(gotScore-tc.Score.Value) > 1e-6 {
				t.Fatalf("score=%g want=%g", gotScore, tc.Score.Value)
			}

			// Noul is not a public request kind in the current Go envelope.
			// Compare only its nine-bin softmax + expected-rating primitive;
			// this test does not claim upstream Noul wire or prompt support.
			bins := make([]Label, 9)
			for i := range bins {
				bins[i] = Label{TokenID: i, ID: fixture.Branches[2].Labels[i], Logit: tc.Logits[2][i], Value: float64(i + 1)}
			}
			rating, _, err := Ordinal(bins)
			if err != nil {
				t.Fatal(err)
			}
			mapped := math.Min(.99, math.Max(.01, .01+(rating-1)*(.98/8)))
			if math.Abs(mapped-tc.Noul.Value) > 1e-6 {
				t.Fatalf("Noul numerical boundary=%g want=%g", mapped, tc.Noul.Value)
			}
		})
	}
	// The independent oracle rejects these malformed branch rows. Native
	// request/branch validation is separate from these numerical primitives.
}

func verifyFixtureSHA(path, want string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != want {
		return fmt.Errorf("fixture SHA-256 mismatch for %s: %s", path, got)
	}
	return nil
}

func checkOracleProbs(t *testing.T, got []float32, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("probability count %d want %d", len(got), len(want))
	}
	for i, p := range got {
		if math.Abs(float64(p)-want[i]) > 1e-6 {
			t.Fatalf("probability[%d]=%g want=%g", i, p, want[i])
		}
	}
}
