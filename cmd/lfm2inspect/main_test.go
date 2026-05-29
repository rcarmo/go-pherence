package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
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
	for _, want := range []string{`"model_type": "lfm2_moe"`, `"conv_layers": 2`, `"attention_layers": 1`, `"runtime_plan"`, `"kv_floats_per_token": 1024`} {
		if !strings.Contains(out, want) {
			t.Fatalf("output missing %s:\n%s", want, out)
		}
	}
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
