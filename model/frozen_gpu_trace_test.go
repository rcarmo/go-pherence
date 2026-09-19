package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestFrozenGPULayerReferenceDiagnostic(t *testing.T) {
	dir, trace := os.Getenv("JEVLIKE_QWEN3_MODEL_DIR"), os.Getenv("JEVLIKE_QWEN3_LAYER_TRACE")
	if dir == "" || trace == "" {
		t.Skip("requires local model/trace")
	}
	var cfg struct {
		Tokens []int `json:"tokens"`
		Width  int   `json:"width"`
		Layers int   `json:"layers"`
	}
	data, err := os.ReadFile(filepath.Join(trace, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(data, &cfg); err != nil {
		t.Fatal(err)
	}
	e, err := NewFrozenGPUEncoder(dir, FrozenGPUOptions{MaxTokens: 512, BudgetBytes: 10 << 30, ReserveBytes: 1 << 30})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	if cfg.Width != e.cfg.HiddenSize || cfg.Layers != len(e.layers) {
		t.Fatal("trace shape mismatch")
	}
	b, h := len(cfg.Tokens), cfg.Width
	read := func(layer int, kind string) []float32 {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(trace, fmt.Sprintf("layer-%02d-%s.f32", layer, kind)))
		if err != nil {
			t.Fatal(err)
		}
		if len(data) != b*h*4 {
			t.Fatal("trace size")
		}
		x := make([]float32, b*h)
		for i := range x {
			x[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
		return x
	}
	for i, tok := range cfg.Tokens {
		if err = e.uploadEmbedding(tok, i); err != nil {
			t.Fatal(err)
		}
	}
	independent := os.Getenv("JEVLIKE_TRACE_INDEPENDENT") == "1"
	for layer := range e.layers {
		if independent {
			if err = e.hidden.GPUBuffer().Upload(read(layer, "input")); err != nil {
				t.Fatal(err)
			}
		}
		if err = e.forwardLayer(layer, b); err != nil {
			t.Fatal(err)
		}
		got := make([]float32, b*h)
		if err = e.hidden.GPUBuffer().Download(got); err != nil {
			t.Fatal(err)
		}
		want := read(layer, "output")
		var mx, sq float64
		worst := 0
		for i, v := range got {
			d := math.Abs(float64(v) - float64(want[i]))
			if d > mx {
				mx = d
				worst = i
			}
			sq += d * d
		}
		t.Logf("layer=%d independent=%v max=%g rms=%g worst=[%d,%d] got=%g want=%g", layer, independent, mx, math.Sqrt(sq/float64(len(got))), worst/h, worst%h, got[worst], want[worst])
	}
}
