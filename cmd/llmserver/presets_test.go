package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseModelPresets(t *testing.T) {
	path := filepath.Join(t.TempDir(), "models.ini")
	data := `# llama.cpp model presets
[qwen-reap]
model = /opt/models/qwen.gguf
threads = 4
batch-size = 512
gpu-layers = 26
ctx-size = 32768
cache-type-k = turbo4
cache-type-v = turbo2

[glm-reap]
model = /opt/models/glm.gguf
ngl = 30
ubatch = 1024
`
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	presets, err := ParseModelPresets(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(presets) != 2 {
		t.Fatalf("got %d presets", len(presets))
	}
	qwen := presets[0]
	if qwen.ID != "qwen-reap" || qwen.Path != "/opt/models/qwen.gguf" || qwen.Threads != 4 || qwen.BatchSize != 512 || qwen.GPULayers != 26 || qwen.CtxSize != 32768 || qwen.CacheTypeK != "turbo4" || qwen.CacheTypeV != "turbo2" {
		t.Fatalf("unexpected qwen preset: %+v", qwen)
	}
	glm := presets[1]
	if glm.ID != "glm-reap" || glm.GPULayers != 30 || glm.BatchSize != 1024 {
		t.Fatalf("unexpected glm preset: %+v", glm)
	}
}

func TestParseModelPresetsRequiresModel(t *testing.T) {
	path := filepath.Join(t.TempDir(), "models.ini")
	if err := os.WriteFile(path, []byte("[broken]\nthreads = 4\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ParseModelPresets(path); err == nil {
		t.Fatal("expected missing model error")
	}
}
