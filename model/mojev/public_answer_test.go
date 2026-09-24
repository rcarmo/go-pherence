package mojev

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"
)

func TestPinnedMoJevPublicAnswers(t *testing.T) {
	for _, pin := range []struct{ path, sha string }{{"../../scripts/mojev_oracle_public_answers.py", "ad1cb546e5e3f15e58869b10cf16aaf0f2b2b120b8fbfe1d5718fc9a9de39ede"}, {"testdata/public_answers.json", "c745a9e1f9b573e7caf02a3969bb9b2ef5dc42f7a818cd043762c0dbb7287d76"}} {
		data, err := os.ReadFile(pin.path)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(data)) != pin.sha {
			t.Fatalf("oracle hash changed: %s", pin.path)
		}
	}
	data, err := os.ReadFile("testdata/public_answers.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema   int    `json:"schema"`
		Revision string `json:"source_revision"`
		ServeSHA string `json:"serve_sha256"`
		FullSHA  string `json:"full_sha256"`
		Cases    []struct {
			Name          string         `json:"name"`
			Kind          string         `json:"kind"`
			Keys          []string       `json:"keys"`
			Options       []string       `json:"options"`
			Order         []int          `json:"sorted_indices"`
			Logits        []float64      `json:"sorted_logits"`
			Probabilities []float64      `json:"probabilities"`
			Answer        map[string]any `json:"answer"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != 1 || fixture.Revision != SourceRevision || fixture.ServeSHA != "7d244785d11eb5cdb490080f24f1307b2c5df8126a6b0858c4f1db341d838c8d" || fixture.FullSHA != "5e2972605c190a511cdfb3a1c01358c4855b229c93a1679a6c569c5b43b07b79" || len(fixture.Cases) != 5 {
		t.Fatal("unexpected oracle provenance")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			before := append([]float64(nil), tc.Logits...)
			got, err := AssembleAnswer(tc.Kind, tc.Keys, tc.Options, tc.Logits)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(before, tc.Logits) {
				t.Fatal("mutated logits")
			}
			if len(got) != len(tc.Answer) || !reflect.DeepEqual(got["type"], tc.Answer["type"]) {
				t.Fatal("wrong answer shape or kind")
			}
			for _, key := range []string{"choice", "noul", "score", "confidence"} {
				if want, ok := tc.Answer[key]; ok {
					if s, ok := want.(string); ok {
						if got[key] != s {
							t.Fatalf("%s = %v want %s", key, got[key], s)
						}
					} else if math.Abs(got[key].(float64)-want.(float64)) > 1e-12 {
						t.Fatalf("%s = %v want %v", key, got[key], want)
					}
				}
			}
			for _, key := range []string{"probabilities", "legend"} {
				if want, ok := tc.Answer[key]; ok {
					switch entries := want.(type) {
					case map[string]any:
						m, ok := got[key].(map[string]float64)
						if ok {
							if len(m) != len(entries) {
								t.Fatalf("%s cardinality: %d want %d", key, len(m), len(entries))
							}
							for label, v := range entries {
								if math.Abs(m[label]-v.(float64)) > 1e-12 {
									t.Fatalf("%s[%s] = %v want %v", key, label, m[label], v)
								}
							}
						} else {
							legend, ok := got[key].(map[string]string)
							if !ok {
								t.Fatal("wrong legend type")
							}
							if len(legend) != len(entries) {
								t.Fatalf("legend cardinality: %d want %d", len(legend), len(entries))
							}
							for label, v := range entries {
								if legend[label] != v {
									t.Fatalf("legend[%s] = %q want %v", label, legend[label], v)
								}
							}
						}
					}
				}
			}
			if m, ok := got["probabilities"].(map[string]float64); ok {
				for _, key := range tc.Keys {
					m[key] = 99
				}
				if len(tc.Logits) > 0 && tc.Logits[0] == 99 {
					t.Fatal("returned map aliases caller")
				}
			}
		})
	}
}

func TestAssembleAnswerRejectsMalformed(t *testing.T) {
	baseKeys := []string{"a", "b"}
	baseOptions := []string{"a: one", "b: two"}
	baseLogits := []float64{1, 2}
	for name, tc := range map[string]struct {
		kind          string
		keys, options []string
		logits        []float64
	}{
		"kind":            {"unknown", baseKeys, baseOptions, baseLogits},
		"missing options": {"choice", baseKeys, nil, baseLogits},
		"short logits":    {"choice", baseKeys, baseOptions, baseLogits[:1]},
		"mismatch keys":   {"choice", baseKeys[:1], baseOptions, baseLogits},
		"empty key":       {"choice", []string{"", "b"}, baseOptions, baseLogits},
		"duplicate key":   {"choice", []string{"a", "a"}, baseOptions, baseLogits},
		"duplicate text":  {"choice", baseKeys, []string{"same", "same"}, baseLogits},
		"nan":             {"choice", baseKeys, baseOptions, []float64{1, math.NaN()}},
		"inf":             {"choice", baseKeys, baseOptions, []float64{1, math.Inf(1)}},
		"noul keys":       {"noul", baseKeys, baseOptions, baseLogits},
		"score order":     {"score", baseKeys, baseOptions, baseLogits},
		"too many":        {"choice", make([]string, 65), make([]string, 65), make([]float64, 65)},
	} {
		t.Run(name, func(t *testing.T) {
			got, err := AssembleAnswer(tc.kind, tc.keys, tc.options, tc.logits)
			if err == nil || got != nil {
				t.Fatalf("accepted malformed answer: %v", got)
			}
		})
	}
	got, err := AssembleAnswer("choice", []string{"z", "a"}, []string{"z: last", "a: first"}, []float64{-300, 300})
	if err != nil || got["choice"] != "z" {
		t.Fatalf("sorted-logit remap failed: %v %v", got, err)
	}
}
