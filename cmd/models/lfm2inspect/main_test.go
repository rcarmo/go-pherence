package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/model/lfm2"
)

func TestInspectMetadataSmoke(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "config.json"), `{
		"model_type":"lfm2_moe",
		"architectures":["Lfm2MoeForCausalLM"],
		"dtype":"bfloat16",
		"vocab_size":128000,
		"hidden_size":2048,
		"num_hidden_layers":3,
		"num_attention_heads":32,
		"num_key_value_heads":8,
		"layer_types":["conv","conv","full_attention"],
		"num_dense_layers":2,
		"num_experts":32,
		"num_experts_per_tok":4,
		"moe_intermediate_size":1792,
		"conv_L_cache":3
	}`)

	out := runInspect(t, "-model", dir, "-json")
	for _, want := range []string{`"model_type": "lfm2_moe"`, `"conv_layers": 2`, `"attention_layers": 1`, `"runtime_plan"`, `"runtime_status"`, `"readiness"`, `"ready_for_execution": false`, `"runtime_implemented": false`, `"cpu_generation_runtime"`, `"kv_floats_per_token": 1024`} {
		if !strings.Contains(out, want) {
			t.Fatalf("output missing %s:\n%s", want, out)
		}
	}
}

func TestInspectReferenceCoverageFixture(t *testing.T) {
	dir := t.TempDir()
	cfg := `{
		"model_type":"lfm2_moe",
		"architectures":["Lfm2MoeForCausalLM"],
		"dtype":"bfloat16",
		"vocab_size":128000,
		"hidden_size":2048,
		"num_hidden_layers":3,
		"num_attention_heads":32,
		"num_key_value_heads":8,
		"layer_types":["conv","conv","full_attention"],
		"num_dense_layers":2,
		"num_experts":32,
		"num_experts_per_tok":4,
		"moe_intermediate_size":1792,
		"conv_L_cache":3
	}`
	writeFile(t, filepath.Join(dir, "config.json"), cfg)
	fixture := filepath.Join(dir, "fixture.json")
	writeFile(t, fixture, `{"config":`+cfg+`,"references":{"tokenization":{"text":"Hello","tokens":[1,2]}},"runtime_request":{"prompt_tokens":2,"max_new_tokens":4,"max_sequence":6,"bytes_per_float":2,"kv_bytes":12288,"conv_state_bytes":24576}}`)

	out := runInspect(t, "-model", dir, "-fixture", fixture, "-json")
	for _, want := range []string{`"reference_coverage"`, `"config_metadata": true`, `"runtime_plan": true`, `"complete_runtime_trace": false`, `"tokenization_fixture"`, `"runtime_request_plan"`, `"max_sequence": 6`, `"kv_bytes": 12288`} {
		if !strings.Contains(out, want) {
			t.Fatalf("output missing %s:\n%s", want, out)
		}
	}
}

func TestRequireRuntimeFailsWhileUnimplemented(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "config.json"), `{"model_type":"lfm2_moe","hidden_size":2048,"num_hidden_layers":1,"num_attention_heads":32,"num_key_value_heads":8,"layer_types":["conv"],"num_experts":32,"num_experts_per_tok":4,"moe_intermediate_size":1792,"conv_L_cache":3}`)
	if out, err := runInspectRaw("-model", dir, "-require-runtime"); err == nil {
		t.Fatalf("expected runtime requirement failure, output:\n%s", out)
	}
}

func TestRequireReadyFailsWhileUnimplemented(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "config.json"), `{"model_type":"lfm2_moe","hidden_size":2048,"num_hidden_layers":1,"num_attention_heads":32,"num_key_value_heads":8,"layer_types":["conv"],"num_experts":32,"num_experts_per_tok":4,"moe_intermediate_size":1792,"conv_L_cache":3}`)
	if out, err := runInspectRaw("-model", dir, "-require-ready"); err == nil {
		t.Fatalf("expected ready requirement failure, output:\n%s", out)
	}
}

func TestRequireNumericParityFailsForPlaceholderFixture(t *testing.T) {
	dir := t.TempDir()
	cfg := `{"model_type":"lfm2_moe","hidden_size":2048,"num_hidden_layers":1,"num_attention_heads":32,"num_key_value_heads":8,"layer_types":["conv"],"num_experts":32,"num_experts_per_tok":4,"moe_intermediate_size":1792,"conv_L_cache":3}`
	writeFile(t, filepath.Join(dir, "config.json"), cfg)
	fixture := filepath.Join(dir, "fixture.json")
	writeFile(t, fixture, `{"config":`+cfg+`,"tensors":{"total":4,"embedding":1,"layers":1,"router":1,"experts":1,"readiness":{"ready":true,"present_required":{"embedding":true,"layers":true,"router":true,"experts":true}}},"references":{"tokenization":{"text":"Hello","tokens":[1]},"first_token":{"token_id":1,"logit_checksum":"pending-transformers"},"conv_layer":{"layer":0,"checksum":"pending-transformers"},"router_topk":{"layer":0,"expert_ids":[0,1,2,3]},"expert_output":{"layer":0,"checksum":"pending-transformers"}},"runtime_request":{"prompt_tokens":1,"max_new_tokens":1,"max_sequence":2,"bytes_per_float":2,"kv_bytes":0,"conv_state_bytes":12288}}`)
	if out, err := runInspectRaw("-model", dir, "-fixture", fixture, "-require-numeric-parity"); err == nil {
		t.Fatalf("expected numeric parity requirement failure, output:\n%s", out)
	}
}

func TestRequireCompleteFixtureFailsForMetadataOnlyFixture(t *testing.T) {
	dir := t.TempDir()
	cfg := `{"model_type":"lfm2_moe","hidden_size":2048,"num_hidden_layers":1,"num_attention_heads":32,"num_key_value_heads":8,"layer_types":["conv"],"num_experts":32,"num_experts_per_tok":4,"moe_intermediate_size":1792,"conv_L_cache":3}`
	writeFile(t, filepath.Join(dir, "config.json"), cfg)
	fixture := filepath.Join(dir, "fixture.json")
	writeFile(t, fixture, cfg)
	if out, err := runInspectRaw("-model", dir, "-fixture", fixture, "-require-complete-fixture"); err == nil {
		t.Fatalf("expected incomplete fixture failure, output:\n%s", out)
	}
}

func TestStrictWithoutTensorMetadataPasses(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "config.json"), `{"model_type":"lfm2_moe","hidden_size":2048,"num_hidden_layers":1,"num_attention_heads":32,"num_key_value_heads":8,"layer_types":["conv"],"num_experts":32,"num_experts_per_tok":4,"moe_intermediate_size":1792,"conv_L_cache":3}`)
	runInspect(t, "-model", dir, "-strict")
}

func runInspect(t *testing.T, args ...string) string {
	t.Helper()
	out, err := runInspectRaw(args...)
	if err != nil {
		t.Fatalf("inspect failed: %v\n%s", err, out)
	}
	return out
}

func runInspectRaw(args ...string) (string, error) {
	cmd := exec.Command(os.Args[0], append([]string{"-test.run=TestHelperProcess", "--"}, args...)...)
	cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
	out, err := cmd.CombinedOutput()
	return string(out), err
}

func TestReportValidStrictInputs(t *testing.T) {
	if !reportValid(report{}) {
		t.Fatal("empty report should be valid")
	}
	if reportValid(report{TensorCoverage: &lfm2.TensorCoverage{Readiness: lfm2.TensorReadiness{Ready: false}}}) {
		t.Fatal("expected tensor readiness failure")
	}
	if reportValid(report{ShapeValidation: &lfm2.TensorShapeValidation{Valid: false, Issues: []string{"bad"}}}) {
		t.Fatal("expected shape validation failure")
	}
	if !reportValid(report{TensorCoverage: &lfm2.TensorCoverage{Readiness: lfm2.TensorReadiness{Ready: true}}, ShapeValidation: &lfm2.TensorShapeValidation{Valid: true}}) {
		t.Fatal("expected valid strict report")
	}
}

func TestHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}
	for i, arg := range os.Args {
		if arg == "--" {
			os.Args = append([]string{os.Args[0]}, os.Args[i+1:]...)
			main()
			return
		}
	}
	os.Exit(2)
}

func writeFile(t *testing.T, path, data string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(data), 0644); err != nil {
		t.Fatal(err)
	}
}
