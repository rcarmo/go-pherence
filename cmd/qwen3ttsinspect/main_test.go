package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/model/qwen3tts"
)

func TestInspectTokenizedPromptSmoke(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "config.json"), `{"tts_model_type":"custom_voice","tts_model_size":"0b6","talker_config":{"hidden_size":1024,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":64}}}`)
	writeFile(t, filepath.Join(dir, "vocab.json"), `{"Hello":9707,"Ġworld":1879}`)
	writeFile(t, filepath.Join(dir, "merges.txt"), "#version: 0.2\n")

	out := runInspect(t, "-model", dir, "-text", "Hello world", "-json")
	for _, want := range []string{`"label": "0.6B CustomVoice"`, `"first_text_id": 9707`, `"runtime_plan"`, `"runtime_status"`, `"runtime_implemented": false`, `"cpu_talker_runtime"`, `"codec_stream"`} {
		if !strings.Contains(out, want) {
			t.Fatalf("output missing %s:\n%s", want, out)
		}
	}
}

func TestInspectReferenceCoverageFixture(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "config.json"), `{"tts_model_type":"custom_voice","tts_model_size":"0b6","talker_config":{"hidden_size":1024,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":64}}}`)
	fixture := filepath.Join(dir, "fixture.json")
	writeFile(t, fixture, `{"name":"probe","variant":"custom_voice","model_size":"0b6","text":"Hello","speaker":"ryan","language":"en","prompt":{"text":[151644,77091,198,151859,151859,151859,151859,151859,151860,9707],"codec":[151860,151861,2050,151862,3061,151859,151860]}}`)

	out := runInspect(t, "-model", dir, "-fixture", fixture, "-json")
	for _, want := range []string{`"reference_coverage"`, `"prompt": true`, `"complete_runtime_trace": false`, `"semantic_token"`} {
		if !strings.Contains(out, want) {
			t.Fatalf("output missing %s:\n%s", want, out)
		}
	}
}

func TestRequireRuntimeFailsWhileUnimplemented(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "config.json"), `{"tts_model_type":"custom_voice","tts_model_size":"0b6","talker_config":{"hidden_size":1024,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":64}}}`)
	if out, err := runInspectRaw("-model", dir, "-require-runtime"); err == nil {
		t.Fatalf("expected runtime requirement failure, output:\n%s", out)
	}
}

func TestRequireCompleteFixtureFailsForPromptOnlyFixture(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "config.json"), `{"tts_model_type":"custom_voice","tts_model_size":"0b6","talker_config":{"hidden_size":1024,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":64}}}`)
	fixture := filepath.Join(dir, "fixture.json")
	writeFile(t, fixture, `{"name":"probe","variant":"custom_voice","model_size":"0b6","text":"Hello","speaker":"ryan","language":"en","prompt":{"text":[151644,77091,198,151859,151859,151859,151859,151859,151860,9707],"codec":[151860,151861,2050,151862,3061,151859,151860]}}`)
	if out, err := runInspectRaw("-model", dir, "-fixture", fixture, "-require-complete-fixture"); err == nil {
		t.Fatalf("expected incomplete fixture failure, output:\n%s", out)
	}
}

func TestStrictWithoutTensorMetadataPasses(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "config.json"), `{"tts_model_type":"custom_voice","tts_model_size":"0b6","talker_config":{"hidden_size":1024,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":64}}}`)
	runInspect(t, "-model", dir, "-strict")
}

func TestConditioningValidationSmoke(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "config.json"), `{"tts_model_type":"base","tts_model_size":"0b6","talker_config":{"hidden_size":1024,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":64}}}`)
	out := runInspect(t, "-model", dir, "-reference-audio", "voice.wav", "-json")
	if !strings.Contains(out, `"conditioning_validation"`) || !strings.Contains(out, `"valid": true`) {
		t.Fatalf("conditioning validation missing:\n%s", out)
	}
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
	if reportValid(report{TensorCoverage: &qwen3tts.TensorCoverage{Readiness: qwen3tts.TensorReadiness{Ready: false}}}) {
		t.Fatal("expected tensor readiness failure")
	}
	if reportValid(report{ShapeValidation: &qwen3tts.TensorShapeValidation{Valid: false, Issues: []string{"bad"}}}) {
		t.Fatal("expected shape validation failure")
	}
	if reportValid(report{ConditioningValidation: &qwen3tts.ConditioningValidation{Valid: false, Error: "bad"}}) {
		t.Fatal("expected conditioning validation failure")
	}
	if !reportValid(report{TensorCoverage: &qwen3tts.TensorCoverage{Readiness: qwen3tts.TensorReadiness{Ready: true}}, ShapeValidation: &qwen3tts.TensorShapeValidation{Valid: true}, ConditioningValidation: &qwen3tts.ConditioningValidation{Valid: true}}) {
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
