package mojev

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"reflect"
	"testing"
)

type groupedTextFixture struct {
	Schema              int    `json:"schema"`
	Policy              string `json:"policy"`
	Revision            string `json:"source_revision"`
	ConfigSHA           string `json:"config_sha256"`
	WeightSHA           string `json:"weights_sha256"`
	ModelSHA            string `json:"modeling_sha256"`
	WeightSize          int64  `json:"weights_size"`
	TransformersVersion string `json:"transformers_version"`
	TorchVersion        string `json:"torch_version"`
	TransformersSHA     string `json:"transformers_qwen_sha256"`

	Row            EncodedRow    `json:"row"`
	Logits         [][]float32   `json:"logits"`
	HiddenSamples  []hiddenGroup `json:"hidden_samples"`
	SiblingChanged changedLogit  `json:"sibling_changed"`
}

type hiddenGroup struct {
	Candidate int         `json:"candidate"`
	Positions []int       `json:"positions"`
	Hidden    [][]float32 `json:"hidden"`
}

type changedLogit struct {
	Candidate int     `json:"candidate"`
	Token     int     `json:"token"`
	Logit     float32 `json:"logit"`
}

func groupedFixture(t *testing.T) groupedTextFixture {
	t.Helper()
	for _, p := range []struct{ path, sha string }{
		{"../../scripts/mojev_oracle_grouped_text.py", "4eeaa8a8c74ca5b3a8134bace2c5939c5ff6e7b8bdccc3999dc03c3ddd1e8428"},
		{"testdata/native_grouped_text.json", "a35a5b2c808793ffc7db9611d981233a9ec5a9db003f2815839d66492d4ca9bd"},
	} {
		b, err := os.ReadFile(p.path)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(b)) != p.sha {
			t.Fatalf("hash mismatch %s", p.path)
		}
	}
	data, err := os.ReadFile("testdata/native_grouped_text.json")
	if err != nil {
		t.Fatal(err)
	}
	var keys map[string]json.RawMessage
	if err = json.Unmarshal(data, &keys); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"schema", "policy", "source_revision", "config_sha256", "weights_sha256", "row", "logits", "hidden_samples", "sibling_changed", "modeling_sha256", "weights_size", "transformers_version", "torch_version", "transformers_qwen_sha256"} {
		if _, ok := keys[key]; !ok {
			t.Fatal("fixture keys")
		}
	}
	if len(keys) != 14 {
		t.Fatal("fixture keys")
	}
	var f groupedTextFixture
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err = dec.Decode(&f); err != nil {
		t.Fatal(err)
	}
	var extra any
	if err = dec.Decode(&extra); err != io.EOF {
		t.Fatal("fixture trailing data")
	}
	base := nativeFixture(t)
	if f.Schema != 1 || f.Policy != "f32-branch-local-positions-full-tree-causal-linear" || f.Revision != SourceRevision || f.ConfigSHA != base.ConfigSHA || f.WeightSHA != base.WeightSHA {
		t.Fatal("fixture provenance")
	}
	if f.TransformersVersion != "5.17.0" || f.TorchVersion != "2.14.0+cu130" || f.WeightSize != 1710234304 || f.ModelSHA != "a8e93f62d92c6748c5d001fef4f9516d6a74b10158d7265f53bab13f1091d458" || f.TransformersSHA != "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b" {
		t.Fatal("implementation pins")
	}
	return f
}

func TestNativeGroupedTextFixture(t *testing.T) {
	f := groupedFixture(t)
	if len(f.Row.State) != 64 || len(f.Row.Questions) != 1 || len(f.Row.Candidates) != 1 || len(f.Row.Questions[0]) != 64 || len(f.Row.Candidates[0]) != 64 || len(f.Logits) != 1 || len(f.Logits[0]) != 64 || len(f.HiddenSamples) != 3 {
		t.Fatal("fixture geometry")
	}
	total := len(f.Row.State) + len(f.Row.Questions[0])
	for i, ids := range f.Row.Candidates[0] {
		if len(ids) != 62 {
			t.Fatalf("candidate geometry %d", i)
		}
		total += len(ids)
	}
	if total != 4096 {
		t.Fatal("fixture size")
	}
	for i, v := range f.Logits[0] {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("nonfinite logit %d", i)
		}
	}
	wantPositions := []int{63, 127, 189}
	wantCandidates := map[int]bool{0: true, 31: true, 63: true}
	seenCandidates := map[int]bool{}
	for _, sample := range f.HiddenSamples {
		if !wantCandidates[sample.Candidate] || seenCandidates[sample.Candidate] {
			t.Fatal("sample candidate")
		}
		seenCandidates[sample.Candidate] = true
		if !reflect.DeepEqual(sample.Positions, wantPositions) || len(sample.Hidden) != len(wantPositions) {
			t.Fatal("sample positions")
		}
		for _, row := range sample.Hidden {
			if len(row) != 1024 {
				t.Fatal("hidden width")
			}
			for _, v := range row {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatal("nonfinite hidden")
				}
			}
		}
	}
	if len(seenCandidates) != 3 {
		t.Fatal("sample count")
	}
	if f.SiblingChanged.Candidate != 63 || f.SiblingChanged.Token != 1234 || math.IsNaN(float64(f.SiblingChanged.Logit)) || math.IsInf(float64(f.SiblingChanged.Logit), 0) {
		t.Fatal("changed sibling")
	}
	if f.SiblingChanged.Logit == f.Logits[0][f.SiblingChanged.Candidate] {
		t.Fatal("changed logit")
	}
}
