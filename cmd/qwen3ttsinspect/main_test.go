package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestInspectTokenizedPromptSmoke(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "config.json"), `{"tts_model_type":"custom_voice","tts_model_size":"0b6","talker_config":{"hidden_size":1024,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":64}}}`)
	writeFile(t, filepath.Join(dir, "vocab.json"), `{"Hello":9707,"Ġworld":1879}`)
	writeFile(t, filepath.Join(dir, "merges.txt"), "#version: 0.2\n")

	out := runInspect(t, "-model", dir, "-text", "Hello world", "-json")
	for _, want := range []string{`"label": "0.6B CustomVoice"`, `"first_text_id": 9707`, `"runtime_plan"`, `"codec_stream"`} {
		if !strings.Contains(out, want) {
			t.Fatalf("output missing %s:\n%s", want, out)
		}
	}
}

func TestStrictWithoutTensorMetadataPasses(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "config.json"), `{"tts_model_type":"custom_voice","tts_model_size":"0b6","talker_config":{"hidden_size":1024,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":64}}}`)
	runInspect(t, "-model", dir, "-strict")
}

func runInspect(t *testing.T, args ...string) string {
	t.Helper()
	cmd := exec.Command(os.Args[0], append([]string{"-test.run=TestHelperProcess", "--"}, args...)...)
	cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("inspect failed: %v\n%s", err, out)
	}
	return string(out)
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
