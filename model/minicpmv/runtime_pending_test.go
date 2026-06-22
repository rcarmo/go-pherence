package minicpmv

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestMetadataAndPendingRuntimeDoNotClaimExecution(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{
  "architectures":["MiniCPMVForCausalLM"],
  "model_type":"minicpmv",
  "text_config":{"model_type":"qwen2","hidden_size":4,"num_hidden_layers":1,"num_attention_heads":1,"vocab_size":100},
  "vision_config":{"model_type":"siglip_vision_model","hidden_size":4,"num_hidden_layers":1,"num_attention_heads":1,"image_size":14,"patch_size":14},
  "resampler_config":{"num_query":1,"num_heads":1,"kv_dim":4}
}`), 0o644); err != nil {
		t.Fatal(err)
	}
	meta, err := LoadMetadata(dir)
	if err != nil {
		t.Fatalf("LoadMetadata: %v", err)
	}
	if meta.Capabilities.EndToEndGeneration || meta.Capabilities.TextRuntimeGeneration || meta.ReadinessReport.RuntimeReady || meta.RuntimePlan.RuntimeReady {
		t.Fatalf("metadata unexpectedly claims runtime execution: caps=%+v readiness=%+v plan=%+v", meta.Capabilities, meta.ReadinessReport, meta.RuntimePlan)
	}
	caps := CurrentCapabilities()
	if caps.EndToEndGeneration || caps.VisionTowerRuntime || caps.ResamplerRuntime || caps.AudioEncoderRuntime {
		t.Fatalf("global capabilities unexpectedly claim runtime: %+v", caps)
	}
	rt := NewPendingRuntimeInterfaces()
	if _, err := rt.Vision.EncodeImage(nil, [4]int{}); !errors.Is(err, ErrRuntimeNotImplemented) {
		t.Fatalf("pending vision runtime err=%v", err)
	}
}
