package minicpmv

import (
	"math"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/config"
)

func tinySigLIPConfig() config.MiniCPMVConfig {
	return config.MiniCPMVConfig{
		Architectures: []string{"MiniCPMOForCausalLM"}, ModelType: "minicpmo",
		HiddenSize: 4, NumHiddenLayers: 1, NumAttentionHeads: 1, NumKeyValueHeads: 1,
		IntermediateSize: 4, VocabSize: 4, NumQuery: 1,
		VisionConfig: &config.MiniCPMVVisionConfig{
			ModelType: "siglip_vision_model", HiddenSize: 2, IntermediateSize: 3,
			NumHiddenLayers: 1, NumAttentionHeads: 1, NumChannels: 3,
			ImageSize: 2, PatchSize: 1, HiddenAct: "gelu_pytorch_tanh", LayerNormEps: 1e-6,
		},
	}
}

func tinySigLIPSource(cfg config.MiniCPMVConfig) fakeTextTensorSource {
	v := cfg.VisionConfig
	src := fakeTextTensorSource{}
	add := func(name string, shape []int, data []float32) {
		src[name] = struct {
			data  []float32
			shape []int
		}{data: data, shape: shape}
	}
	positions := (v.ImageSize / v.PatchSize) * (v.ImageSize / v.PatchSize)
	// Channel 0 maps to hidden 0; channel 1 maps to hidden 1.
	add("vpm.embeddings.patch_embedding.weight", []int{2, 3, 1, 1}, []float32{1, 0, 0, 0, 1, 0})
	add("vpm.embeddings.patch_embedding.bias", []int{2}, []float32{0, 0})
	add("vpm.embeddings.position_embedding.weight", []int{positions, 2}, make([]float32, positions*2))
	add("vpm.post_layernorm.weight", []int{2}, []float32{1, 1})
	add("vpm.post_layernorm.bias", []int{2}, []float32{0, 0})
	prefix := "vpm.encoder.layers.0"
	for _, norm := range []string{"layer_norm1", "layer_norm2"} {
		add(prefix+"."+norm+".weight", []int{2}, []float32{1, 1})
		add(prefix+"."+norm+".bias", []int{2}, []float32{0, 0})
	}
	for _, projection := range []string{"q_proj", "k_proj", "v_proj", "out_proj"} {
		add(prefix+".self_attn."+projection+".weight", []int{2, 2}, make([]float32, 4))
		add(prefix+".self_attn."+projection+".bias", []int{2}, make([]float32, 2))
	}
	add(prefix+".mlp.fc1.weight", []int{3, 2}, make([]float32, 6))
	add(prefix+".mlp.fc1.bias", []int{3}, make([]float32, 3))
	add(prefix+".mlp.fc2.weight", []int{2, 3}, make([]float32, 6))
	add(prefix+".mlp.fc2.bias", []int{2}, make([]float32, 2))
	return src
}

func TestSigLIPVisionCPUEncodeSynthetic(t *testing.T) {
	cfg := tinySigLIPConfig()
	src := tinySigLIPSource(cfg)
	model, err := LoadSigLIPVisionCPU(src, cfg)
	if err != nil {
		t.Fatal(err)
	}
	// BCHW planes produce rows [1,0], [2,0], [0,2], and [3,1].
	pixels := []float32{1, 2, 0, 3, 0, 0, 2, 1, 9, 9, 9, 9}
	out, err := model.EncodeImage(pixels, [4]int{1, 3, 2, 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 8 {
		t.Fatalf("output=%d want 8", len(out))
	}
	want := []float32{1, -1, 1, -1, -1, 1, 1, -1}
	for i := range want {
		if math.Abs(float64(out[i]-want[i])) > 2e-5 {
			t.Fatalf("output[%d]=%g want %g; all=%v", i, out[i], want[i], out)
		}
	}
	// The model owns source tensors and output does not alias internal scratch.
	src["vpm.embeddings.patch_embedding.weight"].data[0] = 99
	out[0] = 99
	again, err := model.EncodeImage(pixels, [4]int{1, 3, 2, 2})
	if err != nil || again[0] == 99 {
		t.Fatalf("owned rerun=%v err=%v", again, err)
	}
}

func TestSigLIPVisionCPUAttentionDeterminism(t *testing.T) {
	cfg := tinySigLIPConfig()
	src := tinySigLIPSource(cfg)
	prefix := "vpm.encoder.layers.0.self_attn."
	for _, projection := range []string{"q_proj", "k_proj", "v_proj", "out_proj"} {
		tensor := src[prefix+projection+".weight"]
		tensor.data = []float32{1, 0, 0, 1}
		src[prefix+projection+".weight"] = tensor
	}
	model, err := LoadSigLIPVisionCPU(src, cfg)
	if err != nil {
		t.Fatal(err)
	}
	pixels := []float32{1, 2, 0, 3, 0, 0, 2, 1, 0, 0, 0, 0}
	first, err := model.EncodeImage(pixels, [4]int{1, 3, 2, 2})
	if err != nil {
		t.Fatal(err)
	}
	second, err := model.EncodeImage(pixels, [4]int{1, 3, 2, 2})
	if err != nil || !closeTextSlice(first, second, 0) {
		t.Fatalf("non-deterministic output first=%v second=%v err=%v", first, second, err)
	}
	for i, value := range first {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("output[%d]=%v", i, value)
		}
	}
}

func TestSigLIPVisionCPURejectsMalformedInputs(t *testing.T) {
	cfg := tinySigLIPConfig()
	if _, err := LoadSigLIPVisionCPU(nil, cfg); err == nil {
		t.Fatal("accepted nil source")
	}
	if _, err := LoadSigLIPVisionCPUFromDir(t.TempDir(), cfg); err == nil {
		t.Fatal("accepted missing checkpoint")
	}
	badPolicy := tinySigLIPConfig()
	badPolicy.VisionConfig.HiddenAct = "gelu"
	if _, err := LoadSigLIPVisionCPU(tinySigLIPSource(badPolicy), badPolicy); err == nil || !strings.Contains(err.Error(), "policy") {
		t.Fatalf("policy error=%v", err)
	}
	badShape := tinySigLIPSource(cfg)
	tensor := badShape["vpm.embeddings.patch_embedding.weight"]
	tensor.shape = []int{2, 1, 1, 3}
	badShape["vpm.embeddings.patch_embedding.weight"] = tensor
	if _, err := LoadSigLIPVisionCPU(badShape, cfg); err == nil || !strings.Contains(err.Error(), "shape=") {
		t.Fatalf("shape error=%v", err)
	}
	model, err := LoadSigLIPVisionCPU(tinySigLIPSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.EncodeImage(make([]float32, 12), [4]int{1, 3, 1, 4}); err == nil {
		t.Fatal("accepted oversized shape")
	}
	small, err := model.EncodeImage([]float32{1, 0, 0}, [4]int{1, 3, 1, 1})
	if err != nil || len(small) != 2 {
		t.Fatalf("dynamic one-patch output=%v err=%v", small, err)
	}
	pixels := make([]float32, 12)
	pixels[1] = float32(math.NaN())
	if _, err := model.EncodeImage(pixels, [4]int{1, 3, 2, 2}); err == nil || !strings.Contains(err.Error(), "non-finite") {
		t.Fatalf("non-finite error=%v", err)
	}
}

var _ VisionTower = (*SigLIPVisionCPU)(nil)
