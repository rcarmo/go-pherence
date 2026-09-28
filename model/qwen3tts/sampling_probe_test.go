package qwen3tts

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// The oracle probe is pinned to Rust/Candle CPU behavior, not interpreted as
// end-to-end sampling parity. In particular, CPU top-p ordering is specific
// to the pinned reference implementation; its sort closure indexes tokens,
// not batches. Generic Go sampling has different tie/cutoff/draw-order rules.
func TestRustCPUSamplingProbePinned(t *testing.T) {
	root := filepath.Join("testdata", "customvoice_0b6_ryan_hello")
	var provenance struct {
		OracleRevision string `json:"oracle_revision"`
		ScriptSHA      string `json:"sampling_probe_oracle_script_sha256"`
		ProbeSHA       string `json:"sampling_probe_sha256"`
	}
	data, err := os.ReadFile(filepath.Join(root, "reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &provenance); err != nil {
		t.Fatal(err)
	}
	if provenance.OracleRevision != "711ceee07cad92673f86de8997bdf54c30caa49f" || provenance.ScriptSHA != "8e32a568e6a4a77d96556d5a9e975611564da57c2116c70806f92590d0e56d64" || provenance.ProbeSHA != "9b4ebc98ba59053e9e01111c7876498f76ef57ec515c0443f5832723e76542d4" {
		t.Fatal("unexpected sampling probe provenance")
	}
	if err := verifyReleasedFile(filepath.Join("..", "..", "scripts", "qwen3tts_oracle_sampling_probe.rs"), provenance.ScriptSHA, 0); err != nil {
		t.Fatal(err)
	}
	if err := verifyReleasedFile(filepath.Join(root, "sampling_probe.json"), provenance.ProbeSHA, 0); err != nil {
		t.Fatal(err)
	}
	data, err = os.ReadFile(filepath.Join(root, "sampling_probe.json"))
	if err != nil {
		t.Fatal(err)
	}
	var probe struct {
		Cases []struct {
			Name     string    `json:"name"`
			Input    []float32 `json:"input_logits"`
			Adjusted []float32 `json:"adjusted_logits"`
			Tokens   []uint32  `json:"tokens"`
		} `json:"cases"`
		Control struct {
			KeptAcoustic float32 `json:"kept_acoustic"`
			KeptEOS      float32 `json:"kept_eos"`
			Suppressed   bool    `json:"suppressed_control_is_negative_infinity"`
		} `json:"tts_control"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		t.Fatal(err)
	}
	want := map[string][]uint32{"topk_ties": {0, 0, 0, 0, 1, 0}, "topp_order": {0, 1, 1, 1, 1, 1}, "greedy": {1, 1, 1, 1, 1, 1}, "repetition": {2, 2, 2, 2, 0, 2}}
	if len(probe.Cases) != len(want) || probe.Control.KeptAcoustic != 2 || probe.Control.KeptEOS != 10 || !probe.Control.Suppressed {
		t.Fatal("unexpected reference control/case schema")
	}
	for _, c := range probe.Cases {
		tokens, ok := want[c.Name]
		if !ok || !reflect.DeepEqual(c.Tokens, tokens) || len(c.Input) != 5 || len(c.Adjusted) != 5 {
			t.Fatalf("unexpected oracle case %s: %v", c.Name, c.Tokens)
		}
		delete(want, c.Name)
	}
	if len(want) != 0 {
		t.Fatalf("missing cases %v", want)
	}
}
