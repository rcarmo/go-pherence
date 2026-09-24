package simplejev

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"
)

type responseRows [][]float32

func (r responseRows) Rows(TextStateRequest) ([][]float32, error) { return r, nil }

func TestPinnedUpstreamPublicResponse(t *testing.T) {
	const scriptSHA = "27b2cb5782bf2784dde915a40bd289166f56c593517f0564f1f045063ff40480"
	const fixtureSHA = "944a495360de991587b37b1cd2f7624d893a9bebfaff8fe9d5238c643348659e"
	if err := verifyFixtureSHA("../../scripts/simplejev_oracle_public_response.py", scriptSHA); err != nil {
		t.Fatal(err)
	}
	if err := verifyFixtureSHA("testdata/upstream_public_response_v1.json", fixtureSHA); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("testdata/upstream_public_response_v1.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema         int             `json:"schema"`
		OracleRepo     string          `json:"oracle_repo"`
		OracleRevision string          `json:"oracle_revision"`
		ScorerSHA      string          `json:"scorer_sha256"`
		Request        json.RawMessage `json:"request"`
		Branches       []struct {
			Branch   string   `json:"branch"`
			Question string   `json:"question"`
			Labels   []string `json:"labels"`
		} `json:"branches"`
		Cases []struct {
			Name     string      `json:"name"`
			Logits   [][]float32 `json:"logits"`
			Response struct {
				Model   string `json:"model"`
				Answers map[string]struct {
					Type          string             `json:"type"`
					Choice        string             `json:"choice"`
					Confidence    float64            `json:"confidence"`
					Probabilities map[string]float64 `json:"probabilities"`
					Score         float64            `json:"score"`
					Legend        map[string]*string `json:"legend"`
					Noul          float64            `json:"noul"`
				} `json:"answers"`
				Usage PublicUsage `json:"usage"`
			} `json:"response"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != 1 || fixture.OracleRepo != "featherless-ai/simple-jev" || fixture.OracleRevision != "b02aa81c915a8193759b3cd33fef74721d6e005b" || fixture.ScorerSHA != "b1b1303ace5218fb40904e7aaf5a1ad01aa3d072c30678d45e921bb0e8613dfa" || len(fixture.Cases) != 2 || len(fixture.Branches) != 3 || fixture.Branches[0].Question != "pick" || !reflect.DeepEqual(fixture.Branches[0].Labels, []string{"A", "B", "C"}) || fixture.Branches[1].Question != "rating" || !reflect.DeepEqual(fixture.Branches[1].Labels, []string{"0", "1", "2"}) || fixture.Branches[2].Question != "truth" || !reflect.DeepEqual(fixture.Branches[2].Labels, []string{"1", "2", "3", "4", "5", "6", "7", "8", "9"}) {
		t.Fatal("unexpected oracle provenance/branch mapping")
	}
	request, err := DecodeTextStateRequest(strings.NewReader(string(fixture.Request)))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			got, err := AssembleTextStateResponse(request, responseRows(tc.Logits), tc.Response.Usage.InputTokens)
			if err != nil {
				t.Fatal(err)
			}
			if got.Model != tc.Response.Model || got.Usage != tc.Response.Usage || len(got.Answers) != len(tc.Response.Answers) {
				t.Fatalf("response envelope mismatch %+v", got)
			}
			for id, want := range tc.Response.Answers {
				a, ok := got.Answers[id]
				if !ok || a.Type != want.Type {
					t.Fatalf("missing or wrong answer %q: %+v", id, a)
				}
				switch id {
				case "pick":
					if a.Choice == nil || *a.Choice != want.Choice || a.Confidence == nil || math.Abs(*a.Confidence-want.Confidence) > 1e-6 {
						t.Fatalf("choice mismatch: %+v", a)
					}
					comparePublicProbs(t, a.Probabilities, want.Probabilities)
				case "rating":
					if a.Score == nil || math.Abs(*a.Score-want.Score) > 1e-6 || a.Confidence == nil || math.Abs(*a.Confidence-want.Confidence) > 1e-6 || !reflect.DeepEqual(a.Legend, want.Legend) {
						t.Fatalf("score mismatch: %+v", a)
					}
					comparePublicProbs(t, a.Probabilities, want.Probabilities)
				case "truth":
					if a.Noul == nil || math.Abs(*a.Noul-want.Noul) > 1e-6 || a.Confidence != nil || a.Probabilities != nil {
						t.Fatalf("Noul mismatch: %+v", a)
					}
				}
			}
		})
	}
}
func comparePublicProbs(t *testing.T, got, want map[string]float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("probability length %d != %d", len(got), len(want))
	}
	for k, v := range want {
		if math.Abs(got[k]-v) > 1e-6 {
			t.Fatalf("probability %s=%g want %g", k, got[k], v)
		}
	}
}

type responseRowsFunc func(TextStateRequest) ([][]float32, error)

func (f responseRowsFunc) Rows(r TextStateRequest) ([][]float32, error) { return f(r) }
func TestTextStateResponseRejectsAndOwns(t *testing.T) {
	const payload = `{"model":"fixture","state":"","questions":{"pick":{"type":"choice","instructions":"","criteria":{"a":null,"b":"other"}},"rating":{"type":"score","instructions":null,"criteria":["low","high"]},"truth":{"type":"noul","instructions":""}}}`
	r, err := DecodeTextStateRequest(strings.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	base := responseRows{{1, 2}, {0, 1}, {0, 0, 0, 0, 0, 0, 0, 0, 0}}
	for name, run := range map[string]func() (PublicResponse, error){
		"missing": func() (PublicResponse, error) { return AssembleTextStateResponse(r, responseRows{{1, 2}}, 1) },
		"wrong row": func() (PublicResponse, error) {
			return AssembleTextStateResponse(r, responseRows{{1, 2}, {0}, {0, 0, 0, 0, 0, 0, 0, 0, 0}}, 1)
		},
		"nonfinite": func() (PublicResponse, error) {
			return AssembleTextStateResponse(r, responseRows{{1, 2}, {0, float32(math.NaN())}, {0, 0, 0, 0, 0, 0, 0, 0, 0}}, 1)
		},
		"nil":       func() (PublicResponse, error) { return AssembleTextStateResponse(r, nil, 1) },
		"bad usage": func() (PublicResponse, error) { return AssembleTextStateResponse(r, base, -1) },
		"provider failure": func() (PublicResponse, error) {
			return AssembleTextStateResponse(r, responseRowsFunc(func(TextStateRequest) ([][]float32, error) { return nil, errors.New("failed") }), 1)
		},
		"raw logits": func() (PublicResponse, error) {
			changed := r
			changed.RawLogits = true
			return AssembleTextStateResponse(changed, base, 1)
		},
	} {
		t.Run(name, func(t *testing.T) {
			got, err := run()
			if err == nil || got.Answers != nil {
				t.Fatalf("accepted invalid response %+v", got)
			}
		})
	}
	before := *r.Questions[0].Choices[1].Description
	result, err := AssembleTextStateResponse(r, responseRowsFunc(func(p TextStateRequest) ([][]float32, error) {
		*p.Questions[0].Choices[1].Description = "corrupt"
		p.Questions[0].ID = "changed"
		return base, nil
	}), 2)
	if err != nil || result.Answers["pick"].Choice == nil || *result.Answers["pick"].Choice != "b" || r.Questions[0].ID != "pick" || *r.Questions[0].Choices[1].Description != before {
		t.Fatalf("provider mutated request/result: %+v %v", result, err)
	}
	base[0][1] = 99
	if *result.Answers["pick"].Choice != "b" {
		t.Fatal("returned answer aliased provider row")
	}
}
