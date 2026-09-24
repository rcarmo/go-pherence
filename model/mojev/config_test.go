package mojev

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"
)

func pinnedConfig(t *testing.T) []byte {
	t.Helper()
	data, err := os.ReadFile("testdata/mojev_config.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 5935 || fmt.Sprintf("%x", sha256.Sum256(data)) != "1b3fb0dd8ae5a1e334b31bae1cfb2e8212231bd549884819304f29334313f4c9" {
		t.Fatal("MoJev config provenance changed")
	}
	return data
}

func TestPinnedMoJevConfigShapeBoundary(t *testing.T) {
	data := pinnedConfig(t)
	layout, err := ReadConfig(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if layout.RuntimeReady || layout.Rank != 512 || layout.ContextTokens != 16384 || layout.Hidden != 1024 || layout.TextLayers != 24 || layout.LinearLayers != 18 || layout.FullLayers != 6 || layout.Options != 64 {
		t.Fatalf("wrong layout %+v", layout)
	}
	if !reflect.DeepEqual(layout.Linear.QKV, []int{1024, 6144}) || !reflect.DeepEqual(layout.Linear.Gate, []int{1024, 2048}) || !reflect.DeepEqual(layout.Linear.Conv1D, []int{4, 6144}) || !reflect.DeepEqual(layout.Full.QProj, []int{4096, 1024}) || !reflect.DeepEqual(layout.Full.KProj, []int{512, 1024}) || !reflect.DeepEqual(layout.Full.OProj, []int{1024, 2048}) {
		t.Fatalf("unexpected Qwen3.5 shapes %+v", layout)
	}
	layout.Linear.QKV[0] = 0
	again, err := ReadConfig(bytes.NewReader(data))
	if err != nil || again.Linear.QKV[0] != 1024 {
		t.Fatalf("metadata result retained writable scratch: %+v %v", again, err)
	}
}

// BenchmarkMoJevReadConfig measures a metadata-only, cold-parse operation;
// model weights, tokenizer and inference are deliberately excluded.
func BenchmarkMoJevReadConfig(b *testing.B) {
	data, err := os.ReadFile("testdata/mojev_config.json")
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		layout, err := ReadConfig(bytes.NewReader(data))
		if err != nil || layout.RuntimeReady {
			b.Fatalf("read config: %+v %v", layout, err)
		}
	}
}

func TestMoJevConfigRejectsDifferentTopology(t *testing.T) {
	data := pinnedConfig(t)
	var original map[string]any
	if err := json.Unmarshal(data, &original); err != nil {
		t.Fatal(err)
	}
	clone := func() map[string]any {
		var v map[string]any
		if err := json.Unmarshal(data, &v); err != nil {
			t.Fatal(err)
		}
		return v
	}
	encoder := func(v map[string]any) map[string]any { return v["encoder_config"].(map[string]any) }
	text := func(v map[string]any) map[string]any { return encoder(v)["text_config"].(map[string]any) }
	cases := map[string]func(map[string]any){
		"model type":    func(v map[string]any) { v["model_type"] = "qwen3_5" },
		"encoder type":  func(v map[string]any) { encoder(v)["model_type"] = "qwen3_5_text" },
		"rank":          func(v map[string]any) { v["rank"] = 511 },
		"context":       func(v map[string]any) { v["context_tokens"] = 0 },
		"head dtype":    func(v map[string]any) { v["dtype"] = "float32" },
		"encoder dtype": func(v map[string]any) { encoder(v)["dtype"] = "float32" },
		"hidden":        func(v map[string]any) { text(v)["hidden_size"] = 768 },
		"layers":        func(v map[string]any) { text(v)["num_hidden_layers"] = 23 },
		"pattern":       func(v map[string]any) { text(v)["layer_types"].([]any)[3] = "linear_attention" },
		"linear width":  func(v map[string]any) { text(v)["linear_value_head_dim"] = 64 },
		"full heads":    func(v map[string]any) { text(v)["num_attention_heads"] = 4 },
		"vision":        func(v map[string]any) { encoder(v)["vision_config"].(map[string]any)["out_hidden_size"] = 768 },
		"schema option": func(v map[string]any) {
			v["schema"].(map[string]any)["fields"].([]any)[0].(map[string]any)["options"].([]any)[8] = "slot-7"
		},
		"schema kind": func(v map[string]any) {
			v["schema"].(map[string]any)["fields"].([]any)[0].(map[string]any)["kind"] = "score"
		},
	}
	for name, change := range cases {
		t.Run(name, func(t *testing.T) {
			v := clone()
			change(v)
			other, err := json.Marshal(v)
			if err != nil {
				t.Fatal(err)
			}
			got, err := ReadConfig(bytes.NewReader(other))
			if err == nil || got.RuntimeReady {
				t.Fatalf("accepted altered topology %+v", got)
			}
		})
	}
	if _, err := ReadConfig(nil); err == nil {
		t.Fatal("accepted nil reader")
	}
	for _, value := range []string{"", `{"rank":512}`, strings.Repeat(" ", MaxConfigBytes+1), `{"model_type":"mojev-scorer","rank":NaN}`} {
		if _, err := ReadConfig(strings.NewReader(value)); err == nil {
			t.Fatalf("accepted invalid config %q", value[:min(40, len(value))])
		}
	}
}
