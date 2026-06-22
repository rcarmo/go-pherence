package minicpmv

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadMetadataSyntheticConfig(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "config.json"), `{
  "architectures":["MiniCPMOForCausalLM"],
  "model_type":"minicpm-o",
  "text_config":{"model_type":"qwen2","hidden_size":3584,"num_hidden_layers":28,"num_attention_heads":28,"num_key_value_heads":4,"intermediate_size":18944,"vocab_size":151666},
  "vision_config":{"model_type":"siglip_vision_model","hidden_size":1152,"num_hidden_layers":27,"num_attention_heads":16,"image_size":448,"patch_size":14},
  "resampler_config":{"num_query":64,"num_heads":28,"kv_dim":1152}
}`)
	writeFile(t, filepath.Join(dir, "preprocessor_config.json"), `{"image_processor_type":"SiglipImageProcessor","size":{"height":448,"width":448},"patch_size":14,"do_resize":true,"do_rescale":true,"do_normalize":true}`)
	writeFile(t, filepath.Join(dir, "tokenizer.json"), `{"added_tokens":[{"id":151640,"content":"<im_start>"},{"id":151641,"content":"<im_end>"},{"id":151642,"content":"<im_patch>"},{"id":151643,"content":"<image>"}],"model":{"vocab":{}}}`)
	meta, err := LoadMetadata(dir)
	if err != nil {
		t.Fatalf("LoadMetadata: %v", err)
	}
	if meta.Summary.ModelType != "minicpm-o" || meta.Processor == nil || meta.Tokenizer == nil || meta.SpecialTokenIDs == nil {
		t.Fatalf("missing loaded metadata: %+v", meta)
	}
	if meta.SpecialTokenIDs.ImagePatch != 151642 || meta.VisionPlan.PatchGrid != 32 || !meta.RuntimePlan.ConfigReady || !meta.RuntimePlan.ProcessorReady || !meta.RuntimePlan.SpecialTokensReady {
		t.Fatalf("bad aggregate metadata: %+v", meta)
	}
	if err := meta.RequireRuntimeReady(); err == nil {
		t.Fatalf("RequireRuntimeReady should fail while tensor execution is pending")
	}
}

func TestLoadMetadataRejectsUnsupported(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "config.json"), `{"architectures":["Other"],"model_type":"other"}`)
	if _, err := LoadMetadata(dir); err == nil {
		t.Fatalf("expected unsupported config to fail")
	}
}

func writeFile(t *testing.T, path, text string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		t.Fatal(err)
	}
}
