package mojev

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"
)

// The fixture localises an existing upstream failure; this test must not be
// interpreted as evidence that the native Go encoder enforces tree isolation.
func TestReleasedIsolationLayers(t *testing.T) {
	for _, pin := range []struct{ path, hash string }{
		{"../../scripts/mojev_oracle_isolation_layers.py", "e820c3a915c9db7eddd34c0003c2a93fd835a8e5a3b35809b038ff0d996752c1"},
		{"testdata/isolation_layers.json", "844379006834b09a1db7606bc0b94f3cd5613ee169192cd66731ddcf45af1f5b"},
	} {
		data, err := os.ReadFile(pin.path)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(data)) != pin.hash {
			t.Fatalf("audit hash changed: %s", pin.path)
		}
	}
	data, err := os.ReadFile("testdata/isolation_layers.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema          int         `json:"schema"`
		Revision        string      `json:"source_revision"`
		ModelSHA        string      `json:"modeling_sha256"`
		Transformers    string      `json:"transformers_version"`
		TransformersSHA string      `json:"transformers_qwen_sha256"`
		ConfigSHA       string      `json:"config_sha256"`
		WeightsSHA      string      `json:"weights_sha256"`
		WeightsSize     int64       `json:"weights_size"`
		Position        int         `json:"changed_position"`
		TokenID         int         `json:"changed_token_id"`
		Base            [][]float64 `json:"base_logits"`
		Changed         [][]float64 `json:"changed_logits"`
		Layers          []struct {
			Layer int                `json:"layer"`
			Kind  string             `json:"kind"`
			Spans map[string]float64 `json:"max_abs_by_span"`
		} `json:"layers"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != 1 || fixture.Revision != SourceRevision || fixture.ModelSHA != "a8e93f62d92c6748c5d001fef4f9516d6a74b10158d7265f53bab13f1091d458" || fixture.Transformers != "5.17.0" || fixture.TransformersSHA != "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b" || fixture.ConfigSHA != "1b3fb0dd8ae5a1e334b31bae1cfb2e8212231bd549884819304f29334313f4c9" || fixture.WeightsSHA != "eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50" || fixture.WeightsSize != 1710234304 || fixture.Position != 7 || fixture.TokenID != 123 || len(fixture.Layers) != 24 {
		t.Fatal("unexpected layer-audit provenance")
	}
	if len(fixture.Base) != 2 || len(fixture.Changed) != 2 || len(fixture.Base[1]) != 2 || len(fixture.Changed[1]) != 2 || math.Abs(fixture.Changed[1][0]-fixture.Base[1][0]) < 0.1 {
		t.Fatal("layer probe lost released logit leak")
	}
	for i, layer := range fixture.Layers {
		want := "linear_attention"
		if i%4 == 3 {
			want = "full_attention"
		}
		if layer.Layer != i || layer.Kind != want || len(layer.Spans) != 7 {
			t.Fatalf("layer %d wrong topology", i)
		}
		for _, name := range []string{"state", "q0", "a0", "a1", "q1", "b0", "b1"} {
			v, ok := layer.Spans[name]
			if !ok || math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
				t.Fatalf("layer %d bad span %s", i, name)
			}
			if (name == "state" || name == "q0" || name == "a0") && v != 0 {
				t.Fatalf("layer %d unexpected earlier-branch movement %s=%g", i, name, v)
			}
		}
	}
	first := fixture.Layers[0].Spans
	if first["a1"] < 0.5 || first["q1"] < 0.07 || first["b0"] < 0.015 || first["b1"] < 0.008 {
		t.Fatalf("first linear layer no longer exposes cross-question influence: %v", first)
	}
	last := fixture.Layers[23].Spans
	if last["q1"] < 1 || last["b0"] < 1 || last["b1"] < 1 {
		t.Fatalf("final layer no longer carries cross-question influence: %v", last)
	}
}
