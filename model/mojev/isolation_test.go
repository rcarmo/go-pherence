package mojev

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"
)

// TestReleasedIsolationAudit records a released-model counterexample. Passing
// here means the isolation regression is detected, not that the Go encoder is
// qualified or the upstream hybrid attention has been repaired.
func TestReleasedIsolationAudit(t *testing.T) {
	for _, pin := range []struct{ path, sha string }{
		{"../../scripts/mojev_oracle_released_isolation.py", "0ce3b458f4799d7ef5350e162955a8770e7d192abc310569b07708b5c978ed6f"},
		{"testdata/released_isolation.json", "7084e08ddf96a7324fb6798354a1a57f97edd9b416a7b5b0efc8990dd78ec35d"},
	} {
		data, err := os.ReadFile(pin.path)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(data)) != pin.sha {
			t.Fatalf("audit hash changed: %s", pin.path)
		}
	}
	data, err := os.ReadFile("testdata/released_isolation.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema          int    `json:"schema"`
		Revision        string `json:"source_revision"`
		ModelSHA        string `json:"modeling_sha256"`
		Transformers    string `json:"transformers_version"`
		TransformersSHA string `json:"transformers_qwen_sha256"`
		ConfigSHA       string `json:"config_sha256"`
		WeightsSHA      string `json:"weights_sha256"`
		WeightsSize     int64  `json:"weights_size"`
		IDs             []int  `json:"base_ids"`
		Changes         []struct {
			Name     string `json:"name"`
			Position int    `json:"position"`
			TokenID  int    `json:"token_id"`
		} `json:"changes"`
		Logits struct {
			Base    [][]float64 `json:"base"`
			Sibling [][]float64 `json:"sibling_candidate"`
			Other   [][]float64 `json:"other_question"`
		} `json:"logits"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != 1 || fixture.Revision != SourceRevision || fixture.ModelSHA != "a8e93f62d92c6748c5d001fef4f9516d6a74b10158d7265f53bab13f1091d458" || fixture.Transformers != "5.17.0" || fixture.TransformersSHA != "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b" || fixture.ConfigSHA != "1b3fb0dd8ae5a1e334b31bae1cfb2e8212231bd549884819304f29334313f4c9" || fixture.WeightsSHA != "eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50" || fixture.WeightsSize != 1710234304 || len(fixture.IDs) != 12 || len(fixture.Changes) != 2 || fixture.Changes[0].Name != "sibling_candidate" || fixture.Changes[0].Position != 7 || fixture.Changes[1].Name != "other_question" || fixture.Changes[1].Position != 8 {
		t.Fatal("unexpected isolation provenance")
	}
	if len(fixture.Logits.Base) != 2 || len(fixture.Logits.Sibling) != 2 || len(fixture.Logits.Other) != 2 {
		t.Fatal("invalid logit fields")
	}
	for _, rows := range [][][]float64{fixture.Logits.Base, fixture.Logits.Sibling, fixture.Logits.Other} {
		for _, row := range rows {
			if len(row) != 2 {
				t.Fatal("invalid candidate logits")
			}
			for _, v := range row {
				if math.IsInf(v, 0) || math.IsNaN(v) {
					t.Fatal("nonfinite released logit")
				}
			}
		}
	}
	base, sibling, other := fixture.Logits.Base, fixture.Logits.Sibling, fixture.Logits.Other
	if math.Abs(sibling[0][0]-base[0][0]) > 1e-7 || math.Abs(other[0][0]-base[0][0]) > 1e-7 {
		t.Fatal("unexpected first-candidate movement")
	}
	if math.Abs(sibling[1][0]-base[1][0]) < 0.1 || math.Abs(sibling[1][1]-base[1][1]) < 0.1 {
		t.Fatalf("missing cross-question leakage: base=%v sibling=%v", base[1], sibling[1])
	}
	if math.Abs(other[1][0]-base[1][0]) < 0.04 || math.Abs(other[0][1]-base[0][1]) > 1e-7 {
		t.Fatal("unexpected other-question response")
	}
}
