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
		"pipeline_plan":                    "../model/qwen3tts/pipeline.go",
		"capability_validation":            "../model/qwen3tts/capabilities.go",
		"speaker_language_compatibility":   "../model/qwen3tts/speaker_language.go",
		"speaker_encoder_layout":           "../model/qwen3tts/speaker_encoder.go",
		"runtime_sizing_plan":              "../model/qwen3tts/shapes.go",
		"runtime_request_validation":       "../model/qwen3tts/runtime_request.go",
		"runtime_request_fixture":          "../model/qwen3tts/testdata/customvoice_reference_placeholder.json",
		"runtime_request_inspector":        "../cmd/qwen3ttsinspect/main.go",
		"runtime_stage_interfaces":         "../model/qwen3tts/runtime_interfaces.go",
		"runtime_status_reporting":         "../model/qwen3tts/runtime_status.go",
		"runtime_readiness_report":         "../model/qwen3tts/readiness.go",
		"runtime_readiness_gate":           "../cmd/qwen3ttsinspect/main.go",
		"execution_readiness_gate":         "../cmd/qwen3ttsinspect/main.go",
		"attention_layout":                 "../model/qwen3tts/attention_layout.go",
		"ffn_layout":                       "../model/qwen3tts/ffn_layout.go",
		"semantic_token_layout":            "../model/qwen3tts/semantic.go",
		"acoustic_frame_layout":            "../model/qwen3tts/frame.go",
		"code_predictor_head_layout":       "../model/qwen3tts/code_predictor_heads.go",
		"decoder_input_layout":             "../model/qwen3tts/decoder_input.go",
		"waveform_layout":                  "../model/qwen3tts/waveform.go",
		"tensor_shape_validation":          "../model/qwen3tts/tensor_shape_validation.go",
		"strict_inspector_mode":            "../cmd/qwen3ttsinspect/main.go",
		"fixture_coverage_make_target":     "../Makefile",
		"reference_coverage_reporting":     "../model/qwen3tts/fixtures.go",
		"placeholder_reference_tracking":   "../model/qwen3tts/fixtures.go",
		"numeric_parity_readiness_gate":    "../cmd/qwen3ttsinspect/main.go",
		"prompt_fixture_scaffold":          "../model/qwen3tts/testdata/customvoice_prompt_fixture.json",
		"semantic_token_reference_fixture": "../model/qwen3tts/testdata/customvoice_reference_placeholder.json",
		"acoustic_frame_reference_fixture": "../model/qwen3tts/testdata/customvoice_reference_placeholder.json",
		"decoded_wav_reference_fixture":    "../model/qwen3tts/testdata/customvoice_reference_placeholder.json",
		"customvoice_prompt_builder":       "../model/qwen3tts/prompt.go",
		"prefill_layout":                   "../model/qwen3tts/prefill.go",
		"talker_input_layout":              "../model/qwen3tts/talker_input.go",
		"prompt_runtime_layout":            "../model/qwen3tts/prompt_runtime.go",
		"embedding_layout":                 "../model/qwen3tts/embedding_layout.go",
	} {
		assertCoverageFile(t, "qwen3_tts", qwen, key, path)
	}
	lfm2 := m.Families["lfm2_moe"].Coverage
	if !lfm2["execution_role_plan"] || lfm2["cpu_generation_runtime"] {
		t.Fatalf("unexpected lfm2_moe coverage: %+v", lfm2)
	}
	for key, path := range map[string]string{
		"layer_schedule":                       "../model/lfm2/schedule.go",
		"execution_role_plan":                  "../model/lfm2/execution.go",
		"routing_plan":                         "../model/lfm2/routing.go",
		"router_layout":                        "../model/lfm2/router_layout.go",
		"ffn_layout":                           "../model/lfm2/ffn_layout.go",
		"norm_layout":                          "../model/lfm2/norm.go",
		"embedding_layout":                     "../model/lfm2/embedding_layout.go",
		"conv_state_layout":                    "../model/lfm2/conv_state.go",
		"conv_projection_layout":               "../model/lfm2/conv_projection.go",
		"attention_kv_layout":                  "../model/lfm2/attention_kv.go",
		"attention_projection_layout":          "../model/lfm2/attention_projection.go",
		"context_layout":                       "../model/lfm2/context.go",
		"rope_layout":                          "../model/lfm2/rope.go",
		"runtime_state_sizing":                 "../model/lfm2/state.go",
		"runtime_request_validation":           "../model/lfm2/runtime_request.go",
		"runtime_request_fixture":              "../model/lfm2/testdata/lfm25_8b_a1b_reference_placeholder.json",
		"runtime_request_inspector":            "../cmd/lfm2inspect/main.go",
		"runtime_stage_interfaces":             "../model/lfm2/runtime_interfaces.go",
		"runtime_status_reporting":             "../model/lfm2/runtime_status.go",
		"runtime_readiness_report":             "../model/lfm2/readiness.go",
		"runtime_readiness_gate":               "../cmd/lfm2inspect/main.go",
		"execution_readiness_gate":             "../cmd/lfm2inspect/main.go",
		"tensor_shape_validation":              "../model/lfm2/tensor_shape_validation.go",
		"strict_inspector_mode":                "../cmd/lfm2inspect/main.go",
		"fixture_coverage_make_target":         "../Makefile",
		"reference_coverage_reporting":         "../model/lfm2/fixtures.go",
		"placeholder_reference_tracking":       "../model/lfm2/fixtures.go",
		"numeric_parity_readiness_gate":        "../cmd/lfm2inspect/main.go",
		"metadata_fixture":                     "../model/lfm2/testdata/lfm25_8b_a1b_metadata.json",
		"tokenization_reference_fixture":       "../model/lfm2/testdata/lfm25_8b_a1b_reference_placeholder.json",
		"first_token_logits_reference_fixture": "../model/lfm2/testdata/lfm25_8b_a1b_reference_placeholder.json",
		"conv_layer_reference_fixture":         "../model/lfm2/testdata/lfm25_8b_a1b_reference_placeholder.json",
		"attention_layer_reference_fixture":    "../model/lfm2/testdata/lfm25_8b_a1b_reference_placeholder.json",
		"router_topk_reference_fixture":        "../model/lfm2/testdata/lfm25_8b_a1b_reference_placeholder.json",
		"expert_output_reference_fixture":      "../model/lfm2/testdata/lfm25_8b_a1b_reference_placeholder.json",
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
