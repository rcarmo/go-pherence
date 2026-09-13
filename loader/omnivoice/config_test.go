package omnivoice

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestLoadConfigAndExpectedShapes(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	if err := os.WriteFile(cfgPath, []byte(sampleConfigJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfig(dir)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	shapes := ExpectedShapes(cfg)
	if len(shapes) != 16 {
		t.Fatalf("ExpectedShapes len=%d want 16", len(shapes))
	}
	assertShape(t, shapes["audio_embeddings.weight"], []int64{10, 4})
	assertShape(t, shapes["audio_heads.weight"], []int64{10, 4})
	assertShape(t, shapes["llm.layers.0.self_attn.q_proj.weight"], []int64{6, 4})
	assertShape(t, shapes["llm.layers.0.self_attn.o_proj.weight"], []int64{4, 6})
}

func TestValidateTensorInfosDetectsIssues(t *testing.T) {
	cfg := sampleConfig(t)
	specs, err := expectedTensorSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	infos := make(map[string]safetensors.TensorInfo, len(specs)+1)
	for name, spec := range specs {
		infos[name] = safetensors.TensorInfo{DType: spec.DType, Shape: int64sToInts(t, spec.Shape)}
	}
	delete(infos, "llm.norm.weight")
	info := infos["llm.layers.0.self_attn.q_proj.weight"]
	info.DType = "I32"
	infos["llm.layers.0.self_attn.q_proj.weight"] = info
	info = infos["audio_heads.weight"]
	info.Shape = []int{11, 4}
	infos["audio_heads.weight"] = info
	infos["unexpected.weight"] = safetensors.TensorInfo{DType: "F16", Shape: []int{1}}

	meta := ValidateTensorInfos(cfg, infos)
	if meta.Valid {
		t.Fatal("ValidateTensorInfos reported valid for malformed infos")
	}
	if len(meta.Missing) != 1 || meta.Missing[0] != "llm.norm.weight" {
		t.Fatalf("Missing=%v", meta.Missing)
	}
	if len(meta.Unexpected) != 1 || meta.Unexpected[0] != "unexpected.weight" {
		t.Fatalf("Unexpected=%v", meta.Unexpected)
	}
	if len(meta.Issues) != 2 {
		t.Fatalf("Issues=%v", meta.Issues)
	}
}

func TestValidateCheckpointSynthetic(t *testing.T) {
	cfg := sampleConfig(t)
	specs, err := expectedTensorSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "model.safetensors")
	writeSyntheticSafetensors(t, path, specs)

	meta, err := ValidateCheckpoint(path, cfg)
	if err != nil {
		t.Fatalf("ValidateCheckpoint: %v", err)
	}
	if !meta.Valid {
		t.Fatalf("ValidateCheckpoint valid=false: %+v", meta)
	}
	if meta.TensorCount != len(specs) || meta.ExpectedTensorCount != len(specs) {
		t.Fatalf("counts got tensor=%d expected=%d want %d", meta.TensorCount, meta.ExpectedTensorCount, len(specs))
	}
	if meta.DTypes["F16"] != len(specs)-1 || meta.DTypes["I64"] != 1 {
		t.Fatalf("DTypes=%v", meta.DTypes)
	}
	if meta.DataBytes <= 0 {
		t.Fatalf("DataBytes=%d want > 0", meta.DataBytes)
	}
}

func TestLoadConfigRejectsUnsupportedLayerType(t *testing.T) {
	cfgJSON := strings.Replace(sampleConfigJSON, `"full_attention"`, `"sliding_attention"`, 1)
	cfgPath := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(cfgPath, []byte(cfgJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadConfig(cfgPath); err == nil {
		t.Fatal("LoadConfig accepted unsupported layer type")
	}
}

func sampleConfig(t *testing.T) Config {
	t.Helper()
	cfg, err := LoadConfig(writeTempConfig(t, sampleConfigJSON))
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func writeTempConfig(t *testing.T, contents string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func assertShape(t *testing.T, got, want []int64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("shape len=%v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("shape=%v want %v", got, want)
		}
	}
}

func int64sToInts(t *testing.T, in []int64) []int {
	t.Helper()
	out := make([]int, len(in))
	for i, v := range in {
		out[i] = int(v)
		if int64(out[i]) != v {
			t.Fatalf("shape value %d overflows int", v)
		}
	}
	return out
}

func writeSyntheticSafetensors(t *testing.T, path string, specs map[string]TensorSpec) {
	t.Helper()
	names := make([]string, 0, len(specs))
	for name := range specs {
		names = append(names, name)
	}
	sortStrings(names)

	header := map[string]any{"__metadata__": map[string]any{}}
	offset := 0
	data := make([]byte, 0)
	for _, name := range names {
		spec := specs[name]
		byteLen := specByteLen(t, spec)
		header[name] = map[string]any{
			"dtype":        spec.DType,
			"shape":        spec.Shape,
			"data_offsets": []int{offset, offset + byteLen},
		}
		data = append(data, make([]byte, byteLen)...)
		offset += byteLen
	}
	rawHeader, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, uint64(len(rawHeader)))
	buf = append(buf, rawHeader...)
	buf = append(buf, data...)
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
}

func specByteLen(t *testing.T, spec TensorSpec) int {
	t.Helper()
	numel := int64(1)
	for _, dim := range spec.Shape {
		numel *= dim
	}
	switch spec.DType {
	case "F16":
		return int(numel * 2)
	case "I64":
		return int(numel * 8)
	default:
		t.Fatalf("unsupported test dtype %q", spec.DType)
	}
	return 0
}

func sortStrings(values []string) {
	for i := 0; i < len(values); i++ {
		for j := i + 1; j < len(values); j++ {
			if values[j] < values[i] {
				values[i], values[j] = values[j], values[i]
			}
		}
	}
}

const sampleConfigJSON = `{
  "architectures": ["OmniVoice"],
  "audio_codebook_weights": [8, 8],
  "audio_mask_id": 4,
  "audio_vocab_size": 5,
  "bos_token_id": null,
  "dtype": "float32",
  "eos_token_id": 42,
  "llm_config": {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 11,
    "chunk_size_feed_forward": 0,
    "dtype": "float32",
    "eos_token_id": 42,
    "head_dim": 3,
    "hidden_act": "silu",
    "hidden_size": 4,
    "initializer_range": 0.02,
    "intermediate_size": 12,
    "layer_types": ["full_attention"],
    "max_position_embeddings": 128,
    "max_window_layers": 1,
    "model_type": "qwen3",
    "num_attention_heads": 2,
    "num_hidden_layers": 1,
    "num_key_value_heads": 1,
    "rms_norm_eps": 0.000001,
    "rope_parameters": {
      "rope_theta": 1000000,
      "rope_type": "default"
    },
    "sliding_window": null,
    "tie_word_embeddings": true,
    "use_cache": true,
    "use_sliding_window": false,
    "vocab_size": 10
  },
  "model_type": "omnivoice",
  "num_audio_codebook": 2,
  "pad_token_id": 11,
  "transformers_version": "5.3.0"
}`
