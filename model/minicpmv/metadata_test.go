package minicpmv

import (
	"encoding/binary"
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
  "audio_config":{"model_type":"whisper_encoder","hidden_size":1280,"num_hidden_layers":32,"num_attention_heads":20,"feature_size":128,"num_mel_bins":128,"sampling_rate":16000},
  "resampler_config":{"num_query":64,"num_heads":28,"kv_dim":1152}
}`)
	writeFile(t, filepath.Join(dir, "preprocessor_config.json"), `{"image_processor_type":"SiglipImageProcessor","size":{"height":448,"width":448},"patch_size":14,"do_resize":true,"do_rescale":true,"do_normalize":true}`)
	writeFile(t, filepath.Join(dir, "tokenizer.json"), `{"added_tokens":[{"id":151640,"content":"<im_start>"},{"id":151641,"content":"<im_end>"},{"id":151642,"content":"<im_patch>"},{"id":151643,"content":"<image>"}],"model":{"vocab":{}}}`)
	writeFile(t, filepath.Join(dir, "generation_config.json"), `{"max_new_tokens":256,"do_sample":true,"temperature":0.7,"top_p":0.9}`)
	meta, err := LoadMetadata(dir)
	if err != nil {
		t.Fatalf("LoadMetadata: %v", err)
	}
	if meta.Summary.ModelType != "minicpm-o" || meta.Processor == nil || meta.Tokenizer == nil || meta.Generation == nil || meta.SpecialTokenIDs == nil {
		t.Fatalf("missing loaded metadata: %+v", meta)
	}
	if meta.SpecialTokenIDs.ImagePatch != 151642 || meta.Generation.MaxNewTokens != 256 || meta.VisionPlan.PatchGrid != 32 || !meta.AudioPlan.MetadataReady || meta.AudioPlan.MelBins != 128 || !meta.RuntimePlan.ConfigReady || !meta.RuntimePlan.ProcessorReady || !meta.RuntimePlan.SpecialTokensReady {
		t.Fatalf("bad aggregate metadata: %+v", meta)
	}
	if err := meta.RequireRuntimeReady(); err == nil {
		t.Fatalf("RequireRuntimeReady should fail while tensor execution is pending")
	}
}

func TestLoadMetadataWithExplicitSafetensors(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "config.json"), `{
  "architectures":["MiniCPMVForCausalLM"],
  "model_type":"minicpmv",
  "text_config":{"model_type":"qwen2","hidden_size":4,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"intermediate_size":8,"vocab_size":100},
  "vision_config":{"model_type":"siglip_vision_model","hidden_size":3,"num_hidden_layers":1,"num_attention_heads":1,"image_size":14,"patch_size":14},
  "resampler_config":{"num_query":1,"num_heads":1,"kv_dim":3}
}`)
	st := writeTinySafetensors(t, t.TempDir())
	meta, err := LoadMetadataWithOptions(dir, MetadataOptions{SafetensorsPath: st})
	if err != nil {
		t.Fatalf("LoadMetadataWithOptions: %v", err)
	}
	if meta.SafetensorsPath != st || meta.Tensors == nil || meta.Tensors.Total != 4 || meta.ShapeValidation.Valid != true || meta.ResamplerPlan == nil || !meta.ResamplerPlan.Ready {
		t.Fatalf("bad explicit safetensors metadata: %+v", meta)
	}
}

func TestLoadMetadataWithExplicitSafetensorsMissing(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "config.json"), `{"architectures":["MiniCPMVForCausalLM"],"model_type":"minicpmv","hidden_size":4,"num_hidden_layers":1,"num_attention_heads":1,"vocab_size":100,"num_query":1}`)
	if _, err := LoadMetadataWithOptions(dir, MetadataOptions{SafetensorsPath: filepath.Join(dir, "missing.safetensors")}); err == nil {
		t.Fatalf("expected explicit missing safetensors to fail")
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

func writeTinySafetensors(t *testing.T, dir string) string {
	t.Helper()
	header := `{"llm.model.embed_tokens.weight":{"dtype":"F32","shape":[100,4],"data_offsets":[0,1600]},"vpm.embeddings.patch_embedding.weight":{"dtype":"F32","shape":[3,3,14,14],"data_offsets":[1600,8656]},"resampler.query.weight":{"dtype":"F32","shape":[1,4],"data_offsets":[8656,8672]},"resampler.kv_proj.weight":{"dtype":"F32","shape":[4,3],"data_offsets":[8672,8720]}}`
	data := make([]byte, 8720)
	path := filepath.Join(dir, "model.safetensors")
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(header)))
	payload := append(append(lenBuf[:], []byte(header)...), data...)
	if err := os.WriteFile(path, payload, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}
