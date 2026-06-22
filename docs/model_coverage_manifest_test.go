package docs_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	modelcoverage "github.com/rcarmo/go-pherence/internal/modelcoverage"
)

type manifest = modelcoverage.Manifest
type manifestFamily = modelcoverage.ManifestFamily

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
	for _, name := range []string{"qwen3_tts", "lfm2_moe", "minicpmv"} {
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
		"pipeline_plan":                     "../model/qwen3tts/pipeline.go",
		"pipeline_execution_contract":       "../model/qwen3tts/pipeline_contract.go",
		"capability_validation":             "../model/qwen3tts/capabilities.go",
		"speaker_language_compatibility":    "../model/qwen3tts/speaker_language.go",
		"speaker_encoder_layout":            "../model/qwen3tts/speaker_encoder.go",
		"runtime_sizing_plan":               "../model/qwen3tts/shapes.go",
		"runtime_request_validation":        "../model/qwen3tts/runtime_request.go",
		"runtime_request_fixture":           "../model/qwen3tts/testdata/customvoice_reference_placeholder.json",
		"runtime_request_inspector":         "../cmd/qwen/qwen3ttsinspect/main.go",
		"runtime_stage_interfaces":          "../model/qwen3tts/runtime_interfaces.go",
		"runtime_status_reporting":          "../model/qwen3tts/runtime_status.go",
		"runtime_readiness_report":          "../model/qwen3tts/readiness.go",
		"runtime_readiness_gate":            "../cmd/qwen/qwen3ttsinspect/main.go",
		"execution_readiness_gate":          "../cmd/qwen/qwen3ttsinspect/main.go",
		"attention_layout":                  "../model/qwen3tts/attention_layout.go",
		"ffn_layout":                        "../model/qwen3tts/ffn_layout.go",
		"semantic_token_layout":             "../model/qwen3tts/semantic.go",
		"acoustic_frame_layout":             "../model/qwen3tts/frame.go",
		"code_predictor_head_layout":        "../model/qwen3tts/code_predictor_heads.go",
		"decoder_input_layout":              "../model/qwen3tts/decoder_input.go",
		"decoder_execution_contract":        "../model/qwen3tts/decoder_contract.go",
		"waveform_layout":                   "../model/qwen3tts/waveform.go",
		"tensor_shape_validation":           "../model/qwen3tts/tensor_shape_validation.go",
		"strict_inspector_mode":             "../cmd/qwen/qwen3ttsinspect/main.go",
		"fixture_coverage_make_target":      "../Makefile",
		"reference_coverage_reporting":      "../model/qwen3tts/fixtures.go",
		"coverage_category_reporting":       "../cmd/models/modelcoverage/main.go",
		"coverage_markdown_reporting":       "../cmd/models/modelcoverage/main.go",
		"coverage_markdown_make_target":     "../Makefile",
		"coverage_csv_reporting":            "../cmd/models/modelcoverage/main.go",
		"coverage_percent_reporting":        "../cmd/models/modelcoverage/main.go",
		"coverage_threshold_gate":           "../cmd/models/modelcoverage/main.go",
		"runtime_roadmap_reporting":         "../cmd/models/modelcoverage/main.go",
		"runtime_roadmap_descriptions":      "../cmd/models/modelcoverage/main.go",
		"runtime_roadmap_prerequisites":     "../cmd/models/modelcoverage/main.go",
		"runtime_roadmap_validation_hints":  "../cmd/models/modelcoverage/main.go",
		"runtime_roadmap_json_reporting":    "../cmd/models/modelcoverage/main.go",
		"runtime_roadmap_phase_ordering":    "../cmd/models/modelcoverage/main.go",
		"runtime_roadmap_package_hints":     "../cmd/models/modelcoverage/main.go",
		"runtime_roadmap_package_filter":    "../cmd/models/modelcoverage/main.go",
		"runtime_roadmap_fixture_hints":     "../cmd/models/modelcoverage/main.go",
		"runtime_roadmap_kind_hints":        "../cmd/models/modelcoverage/main.go",
		"runtime_roadmap_kind_filter":       "../cmd/models/modelcoverage/main.go",
		"next_runtime_reporting":            "../cmd/models/modelcoverage/main.go",
		"next_runtime_json_reporting":       "../cmd/models/modelcoverage/main.go",
		"coverage_snapshot_reporting":       "../cmd/models/modelcoverage/main.go",
		"coverage_snapshot_file":            "model-coverage-snapshot.md",
		"coverage_snapshot_freshness_gate":  "../Makefile",
		"coverage_snapshot_check_target":    "../Makefile",
		"coverage_tmpdir_bootstrap":         "../Makefile",
		"execution_coverage_filter":         "../cmd/models/modelcoverage/main.go",
		"placeholder_reference_tracking":    "../model/qwen3tts/fixtures.go",
		"numeric_parity_readiness_gate":     "../cmd/qwen/qwen3ttsinspect/main.go",
		"prompt_fixture_scaffold":           "../model/qwen3tts/testdata/customvoice_prompt_fixture.json",
		"semantic_token_reference_fixture":  "../model/qwen3tts/testdata/customvoice_reference_placeholder.json",
		"acoustic_frame_reference_fixture":  "../model/qwen3tts/testdata/customvoice_reference_placeholder.json",
		"decoded_wav_reference_fixture":     "../model/qwen3tts/testdata/customvoice_reference_placeholder.json",
		"customvoice_prompt_builder":        "../model/qwen3tts/prompt.go",
		"prefill_layout":                    "../model/qwen3tts/prefill.go",
		"talker_input_layout":               "../model/qwen3tts/talker_input.go",
		"talker_execution_contract":         "../model/qwen3tts/talker_contract.go",
		"code_predictor_execution_contract": "../model/qwen3tts/code_predictor_contract.go",
		"prompt_runtime_layout":             "../model/qwen3tts/prompt_runtime.go",
		"embedding_layout":                  "../model/qwen3tts/embedding_layout.go",
	} {
		assertCoverageFile(t, "qwen3_tts", qwen, key, path)
	}
	minicpmv := m.Families["minicpmv"].Coverage
	if !minicpmv["multimodal_prompt_builder"] || minicpmv["end_to_end_generation"] {
		t.Fatalf("unexpected minicpmv coverage: %+v", minicpmv)
	}
	for key, path := range map[string]string{
		"config_parsing":                     "../loader/config/minicpmv.go",
		"processor_metadata":                 "../loader/config/minicpmv_processor.go",
		"tokenizer_metadata":                 "../loader/config/minicpmv_tokenizer.go",
		"generation_metadata":                "../loader/config/minicpmv_generation.go",
		"image_prompt_builder":               "../model/minicpmv/prompt_text.go",
		"audio_prompt_builder":               "../model/minicpmv/audio_prompt.go",
		"multimodal_prompt_builder":          "../model/minicpmv/multimodal_prompt.go",
		"image_preprocessing":                "../model/minicpmv/image_processor.go",
		"image_file_inspection":              "../model/minicpmv/image_io.go",
		"slice_mode_plan":                    "../model/minicpmv/slice_plan.go",
		"tensor_group_readiness":             "../model/minicpmv/tensors.go",
		"tensor_shape_validation":            "../model/minicpmv/tensor_shape_validation.go",
		"tensor_info_summary":                "../model/minicpmv/tensor_info_summary.go",
		"tensor_header_byte_summary":         "../model/minicpmv/tensor_info_summary.go",
		"text_execution_plan":                "../model/minicpmv/text_plan.go",
		"vision_execution_plan":              "../model/minicpmv/vision_plan.go",
		"resampler_tensor_plan":              "../model/minicpmv/resampler_plan.go",
		"audio_execution_plan":               "../model/minicpmv/audio_plan.go",
		"audio_feature_plan":                 "../model/minicpmv/audio_features.go",
		"audio_feature_frame_estimate":       "../model/minicpmv/audio_features.go",
		"inspect_audio_duration_make_target": "../Makefile",
		"runtime_stage_interfaces":           "../model/minicpmv/runtime_interfaces.go",
		"runtime_readiness_report":           "../model/minicpmv/readiness.go",
		"capability_summary":                 "../model/minicpmv/capabilities.go",
		"runtime_status_constant":            "../model/minicpmv/capabilities.go",
		"support_version_output":             "../model/minicpmv/version.go",
		"version_make_target":                "../Makefile",
		"pending_runtime_steps":              "../model/minicpmv/runtime_status.go",
		"metadata_fixture":                   "../model/minicpmv/testdata/minicpmo_fixture/config.json",
		"metadata_expected_summary":          "../model/minicpmv/testdata/minicpmo_fixture/expected_summary.json",
		"fixture_expected_summary_helper":    "../model/minicpmv/fixture_summary.go",
		"fixture_path_constant":              "../model/minicpmv/fixtures.go",
		"fixture_metadata_helper":            "../model/minicpmv/fixtures.go",
		"fixture_path_make_target":           "../Makefile",
		"fixture_check_make_target":          "../Makefile",
		"embedding_injection_boundary":       "../model/minicpmv/embedding_injection.go",
		"make_inspect_target":                "../Makefile",
		"scaffold_check_make_target":         "../Makefile",
	} {
		assertCoverageFile(t, "minicpmv", minicpmv, key, path)
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
		"moe_execution_contract":               "../model/lfm2/moe_contract.go",
		"ffn_layout":                           "../model/lfm2/ffn_layout.go",
		"norm_layout":                          "../model/lfm2/norm.go",
		"embedding_layout":                     "../model/lfm2/embedding_layout.go",
		"embedding_execution_contract":         "../model/lfm2/embedding_contract.go",
		"conv_state_layout":                    "../model/lfm2/conv_state.go",
		"conv_execution_contract":              "../model/lfm2/conv_contract.go",
		"conv_projection_layout":               "../model/lfm2/conv_projection.go",
		"attention_kv_layout":                  "../model/lfm2/attention_kv.go",
		"attention_execution_contract":         "../model/lfm2/attention_contract.go",
		"attention_projection_layout":          "../model/lfm2/attention_projection.go",
		"context_layout":                       "../model/lfm2/context.go",
		"rope_layout":                          "../model/lfm2/rope.go",
		"runtime_state_sizing":                 "../model/lfm2/state.go",
		"generation_execution_contract":        "../model/lfm2/generation_contract.go",
		"pipeline_execution_contract":          "../model/lfm2/pipeline_contract.go",
		"runtime_request_validation":           "../model/lfm2/runtime_request.go",
		"runtime_request_fixture":              "../model/lfm2/testdata/lfm25_8b_a1b_reference_placeholder.json",
		"runtime_request_inspector":            "../cmd/models/lfm2inspect/main.go",
		"runtime_stage_interfaces":             "../model/lfm2/runtime_interfaces.go",
		"runtime_status_reporting":             "../model/lfm2/runtime_status.go",
		"runtime_readiness_report":             "../model/lfm2/readiness.go",
		"runtime_readiness_gate":               "../cmd/models/lfm2inspect/main.go",
		"execution_readiness_gate":             "../cmd/models/lfm2inspect/main.go",
		"tensor_shape_validation":              "../model/lfm2/tensor_shape_validation.go",
		"strict_inspector_mode":                "../cmd/models/lfm2inspect/main.go",
		"fixture_coverage_make_target":         "../Makefile",
		"reference_coverage_reporting":         "../model/lfm2/fixtures.go",
		"coverage_category_reporting":          "../cmd/models/modelcoverage/main.go",
		"coverage_markdown_reporting":          "../cmd/models/modelcoverage/main.go",
		"coverage_markdown_make_target":        "../Makefile",
		"coverage_csv_reporting":               "../cmd/models/modelcoverage/main.go",
		"coverage_percent_reporting":           "../cmd/models/modelcoverage/main.go",
		"coverage_threshold_gate":              "../cmd/models/modelcoverage/main.go",
		"runtime_roadmap_reporting":            "../cmd/models/modelcoverage/main.go",
		"runtime_roadmap_descriptions":         "../cmd/models/modelcoverage/main.go",
		"runtime_roadmap_prerequisites":        "../cmd/models/modelcoverage/main.go",
		"runtime_roadmap_validation_hints":     "../cmd/models/modelcoverage/main.go",
		"runtime_roadmap_json_reporting":       "../cmd/models/modelcoverage/main.go",
		"runtime_roadmap_phase_ordering":       "../cmd/models/modelcoverage/main.go",
		"runtime_roadmap_package_hints":        "../cmd/models/modelcoverage/main.go",
		"runtime_roadmap_package_filter":       "../cmd/models/modelcoverage/main.go",
		"runtime_roadmap_fixture_hints":        "../cmd/models/modelcoverage/main.go",
		"runtime_roadmap_kind_hints":           "../cmd/models/modelcoverage/main.go",
		"runtime_roadmap_kind_filter":          "../cmd/models/modelcoverage/main.go",
		"next_runtime_reporting":               "../cmd/models/modelcoverage/main.go",
		"next_runtime_json_reporting":          "../cmd/models/modelcoverage/main.go",
		"coverage_snapshot_reporting":          "../cmd/models/modelcoverage/main.go",
		"coverage_snapshot_file":               "model-coverage-snapshot.md",
		"coverage_snapshot_freshness_gate":     "../Makefile",
		"coverage_snapshot_check_target":       "../Makefile",
		"coverage_tmpdir_bootstrap":            "../Makefile",
		"execution_coverage_filter":            "../cmd/models/modelcoverage/main.go",
		"placeholder_reference_tracking":       "../model/lfm2/fixtures.go",
		"numeric_parity_readiness_gate":        "../cmd/models/lfm2inspect/main.go",
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
