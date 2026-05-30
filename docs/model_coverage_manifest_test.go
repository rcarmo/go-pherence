package docs_test

import (
	"encoding/json"
	"os"
	"path/filepath"
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
		for _, pkg := range fam.Packages {
			if st, err := os.Stat(filepath.Join("..", pkg)); err != nil || !st.IsDir() {
				t.Fatalf("%s package path %s missing or not directory: %v", name, pkg, err)
			}
		}
	}
	qwen := m.Families["qwen3_tts"].Coverage
	if !qwen["pipeline_plan"] || qwen["cpu_talker_runtime"] {
		t.Fatalf("unexpected qwen3_tts coverage: %+v", qwen)
	}
	for key, path := range map[string]string{
		"pipeline_plan":              "../model/qwen3tts/pipeline.go",
		"capability_validation":      "../model/qwen3tts/capabilities.go",
		"runtime_sizing_plan":        "../model/qwen3tts/shapes.go",
		"semantic_token_layout":      "../model/qwen3tts/semantic.go",
		"acoustic_frame_layout":      "../model/qwen3tts/frame.go",
		"waveform_layout":            "../model/qwen3tts/waveform.go",
		"tensor_shape_validation":    "../model/qwen3tts/tensor_shape_validation.go",
		"strict_inspector_mode":      "../cmd/qwen3ttsinspect/main.go",
		"prompt_fixture_scaffold":    "../model/qwen3tts/testdata/customvoice_prompt_fixture.json",
		"customvoice_prompt_builder": "../model/qwen3tts/prompt.go",
		"prefill_layout":             "../model/qwen3tts/prefill.go",
	} {
		assertCoverageFile(t, "qwen3_tts", qwen, key, path)
	}
	lfm2 := m.Families["lfm2_moe"].Coverage
	if !lfm2["execution_role_plan"] || lfm2["cpu_generation_runtime"] {
		t.Fatalf("unexpected lfm2_moe coverage: %+v", lfm2)
	}
	for key, path := range map[string]string{
		"layer_schedule":          "../model/lfm2/schedule.go",
		"execution_role_plan":     "../model/lfm2/execution.go",
		"routing_plan":            "../model/lfm2/routing.go",
		"conv_state_layout":       "../model/lfm2/conv_state.go",
		"attention_kv_layout":     "../model/lfm2/attention_kv.go",
		"runtime_state_sizing":    "../model/lfm2/state.go",
		"tensor_shape_validation": "../model/lfm2/tensor_shape_validation.go",
		"strict_inspector_mode":   "../cmd/lfm2inspect/main.go",
		"metadata_fixture":        "../model/lfm2/testdata/lfm25_8b_a1b_metadata.json",
	} {
		assertCoverageFile(t, "lfm2_moe", lfm2, key, path)
	}
}

func assertCoverageFile(t *testing.T, family string, coverage map[string]bool, key, path string) {
	t.Helper()
	if !coverage[key] {
		t.Fatalf("%s coverage %s is not marked true", family, key)
	}
	if st, err := os.Stat(path); err != nil || st.IsDir() {
		t.Fatalf("%s coverage %s file %s missing or not file: %v", family, key, path, err)
	}
}
