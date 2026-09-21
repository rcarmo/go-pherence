package qwen

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func writeLoRATestFile(t *testing.T, path string, tensors map[string]struct {
	shape []int
	data  []float32
}) {
	t.Helper()
	type entry struct {
		DType   string `json:"dtype"`
		Shape   []int  `json:"shape"`
		Offsets []int  `json:"data_offsets"`
	}
	header := map[string]entry{}
	payload := []byte{}
	for name, v := range tensors {
		start := len(payload)
		for _, x := range v.data {
			payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(x))
		}
		header[name] = entry{"F32", v.shape, []int{start, len(payload)}}
	}
	raw, _ := json.Marshal(header)
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(raw)))
	out = append(out, raw...)
	out = append(out, payload...)
	if e := os.WriteFile(path, out, 0600); e != nil {
		t.Fatal(e)
	}
}
func TestLoadQwen35LoRA(t *testing.T) {
	dir := t.TempDir()
	cfg := `{"base_model_name_or_path":"Qwen/Qwen3.5-9B","revision":"pin","r":2,"lora_alpha":4,"lora_bias":false,"task_type":"CAUSAL_LM","target_modules":["q_proj"]}`
	cp := filepath.Join(dir, "adapter_config.json")
	os.WriteFile(cp, []byte(cfg), 0600)
	prefix := "base_model.model.model.language_model.model.layers.0.self_attn.q_proj"
	wp := filepath.Join(dir, "adapter_model.safetensors")
	writeLoRATestFile(t, wp, map[string]struct {
		shape []int
		data  []float32
	}{prefix + ".lora_A.weight": {[]int{2, 3}, []float32{1, 2, 3, 4, 5, 6}}, prefix + ".lora_B.weight": {[]int{4, 2}, []float32{1, 2, 3, 4, 5, 6, 7, 8}}})
	got, c, e := LoadQwen35LoRA(cp, wp)
	if e != nil {
		t.Fatal(e)
	}
	a := got["model.layers.0.self_attn.q_proj"]
	if c.Rank != 2 || a.Scale != 2 || a.A.Shape()[1] != 3 || a.B.Shape()[0] != 4 {
		t.Fatalf("config=%+v adapter=%+v", c, a)
	}
}
func TestLoadQwen35LoRARejectsMalformed(t *testing.T) {
	dir := t.TempDir()
	wp := filepath.Join(dir, "a.safetensors")
	writeLoRATestFile(t, wp, map[string]struct {
		shape []int
		data  []float32
	}{"bad": {[]int{1}, []float32{1}}})
	for i, cfg := range []string{"{", `{"base_model_name_or_path":"x","revision":"y","r":0,"lora_alpha":1,"task_type":"CAUSAL_LM"}`, `{"base_model_name_or_path":"x","revision":"y","r":1,"lora_alpha":1,"task_type":"CAUSAL_LM","target_modules":["q_proj"]}`} {
		cp := filepath.Join(dir, "c.json")
		os.WriteFile(cp, []byte(cfg), 0600)
		if _, _, e := LoadQwen35LoRA(cp, wp); e == nil {
			t.Fatalf("case %d accepted", i)
		}
	}
}
