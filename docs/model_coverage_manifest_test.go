package docs_test

import (
	"encoding/json"
	"os"
	"testing"
)

type manifest struct {
	Version  int                       `json:"version"`
	Families map[string]manifestFamily `json:"families"`
}

type manifestFamily struct {
	Status            string          `json:"status"`
	RuntimeGeneration bool            `json:"runtime_generation"`
	ValidationTarget  string          `json:"validation_target"`
	Packages          []string        `json:"packages"`
	Coverage          map[string]bool `json:"coverage"`
	Commands          []string        `json:"commands"`
}

func TestModelCoverageManifest(t *testing.T) {
	data, err := os.ReadFile("model-coverage-manifest.json")
	if err != nil {
		t.Fatal(err)
	}
	var m manifest
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	if m.Version != 1 {
		t.Fatalf("version=%d", m.Version)
	}
	for _, name := range []string{"qwen3_tts", "lfm2_moe"} {
		fam, ok := m.Families[name]
		if !ok {
			t.Fatalf("missing family %s", name)
		}
		if fam.Status == "" || fam.ValidationTarget == "" || len(fam.Packages) == 0 || len(fam.Commands) == 0 {
			t.Fatalf("incomplete family %s: %+v", name, fam)
		}
		if fam.RuntimeGeneration {
			t.Fatalf("%s should not claim runtime generation yet", name)
		}
		if len(fam.Coverage) == 0 {
			t.Fatalf("%s has no coverage entries", name)
		}
	}
	if !m.Families["qwen3_tts"].Coverage["pipeline_plan"] || m.Families["qwen3_tts"].Coverage["cpu_talker_runtime"] {
		t.Fatalf("unexpected qwen3_tts coverage: %+v", m.Families["qwen3_tts"].Coverage)
	}
	if !m.Families["lfm2_moe"].Coverage["execution_role_plan"] || m.Families["lfm2_moe"].Coverage["cpu_generation_runtime"] {
		t.Fatalf("unexpected lfm2_moe coverage: %+v", m.Families["lfm2_moe"].Coverage)
	}
}
