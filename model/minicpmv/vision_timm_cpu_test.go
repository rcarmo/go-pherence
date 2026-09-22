package minicpmv

import (
	"math"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/config"
)

func tinyTimmVisionConfig() config.MiniCPMVConfig {
	return config.MiniCPMVConfig{
		Architectures: []string{"MiniCPMV"}, ModelType: "minicpmv",
		HiddenSize: 4, NumHiddenLayers: 1, NumAttentionHeads: 1, NumKeyValueHeads: 1,
		IntermediateSize: 4, VocabSize: 4, NumQuery: 1,
		ImageSize: 2, PatchSize: 1, VisionEncoder: "vit_so400m_patch14_siglip_384.webli",
	}
}

func tinyTimmVisionSource(cfg config.MiniCPMVConfig) fakeTextTensorSource {
	// The published preset dimensions are fixed in metadata. For a bounded
	// synthetic graph, override the normalized values through a nested vision
	// config while retaining the official encoder identifier.
	cfg.VisionConfig = &config.MiniCPMVVisionConfig{ModelType: cfg.VisionEncoder, HiddenSize: 2, IntermediateSize: 3, NumHiddenLayers: 1, NumAttentionHeads: 1, NumChannels: 3, ImageSize: 2, PatchSize: 1}
	s := cfg.MiniCPMVSummary()
	src := fakeTextTensorSource{}
	add := func(name string, shape []int, data []float32) {
		src[name] = struct {
			data  []float32
			shape []int
		}{data: data, shape: shape}
	}
	add("vpm.patch_embed.proj.weight", []int{s.VisionHiddenSize, 3, 1, 1}, []float32{1, 0, 0, 0, 1, 0})
	add("vpm.patch_embed.proj.bias", []int{s.VisionHiddenSize}, make([]float32, s.VisionHiddenSize))
	add("vpm.pos_embed", []int{1, 4, s.VisionHiddenSize}, make([]float32, 4*s.VisionHiddenSize))
	add("vpm.norm.weight", []int{s.VisionHiddenSize}, []float32{1, 1})
	add("vpm.norm.bias", []int{s.VisionHiddenSize}, make([]float32, s.VisionHiddenSize))
	prefix := "vpm.blocks.0"
	for _, norm := range []string{"norm1", "norm2"} {
		add(prefix+"."+norm+".weight", []int{s.VisionHiddenSize}, []float32{1, 1})
		add(prefix+"."+norm+".bias", []int{s.VisionHiddenSize}, make([]float32, s.VisionHiddenSize))
	}
	add(prefix+".attn.qkv.weight", []int{3 * s.VisionHiddenSize, s.VisionHiddenSize}, make([]float32, 3*s.VisionHiddenSize*s.VisionHiddenSize))
	add(prefix+".attn.qkv.bias", []int{3 * s.VisionHiddenSize}, make([]float32, 3*s.VisionHiddenSize))
	add(prefix+".attn.proj.weight", []int{s.VisionHiddenSize, s.VisionHiddenSize}, make([]float32, s.VisionHiddenSize*s.VisionHiddenSize))
	add(prefix+".attn.proj.bias", []int{s.VisionHiddenSize}, make([]float32, s.VisionHiddenSize))
	add(prefix+".mlp.fc1.weight", []int{s.VisionIntermediate, s.VisionHiddenSize}, make([]float32, s.VisionIntermediate*s.VisionHiddenSize))
	add(prefix+".mlp.fc1.bias", []int{s.VisionIntermediate}, make([]float32, s.VisionIntermediate))
	add(prefix+".mlp.fc2.weight", []int{s.VisionHiddenSize, s.VisionIntermediate}, make([]float32, s.VisionHiddenSize*s.VisionIntermediate))
	add(prefix+".mlp.fc2.bias", []int{s.VisionHiddenSize}, make([]float32, s.VisionHiddenSize))
	return src
}

func tinyTimmExecutableConfig() config.MiniCPMVConfig {
	cfg := tinyTimmVisionConfig()
	cfg.VisionConfig = &config.MiniCPMVVisionConfig{ModelType: cfg.VisionEncoder, HiddenSize: 2, IntermediateSize: 3, NumHiddenLayers: 1, NumAttentionHeads: 1, NumChannels: 3, ImageSize: 2, PatchSize: 1}
	return cfg
}

func TestTimmSigLIPVisionCPUEncodeSynthetic(t *testing.T) {
	cfg := tinyTimmExecutableConfig()
	src := tinyTimmVisionSource(cfg)
	model, err := LoadTimmSigLIPVisionCPU(src, cfg)
	if err != nil {
		t.Fatal(err)
	}
	src["vpm.patch_embed.proj.weight"].data[0] = 99
	pixels := []float32{1, 2, 0, 3, 0, 0, 2, 1, 9, 9, 9, 9}
	out, err := model.EncodeImage(pixels, [4]int{1, 3, 2, 2})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, -1, 1, -1, -1, 1, 1, -1}
	for i := range want {
		if math.Abs(float64(out[i]-want[i])) > 2e-5 {
			t.Fatalf("output[%d]=%g want %g; all=%v", i, out[i], want[i], out)
		}
	}
	out[0] = 99
	again, err := model.EncodeImage(pixels, [4]int{1, 3, 2, 2})
	if err != nil || again[0] == 99 {
		t.Fatalf("owned rerun=%v err=%v", again, err)
	}
}

func TestTimmSigLIPVisionCPUAttentionDeterminism(t *testing.T) {
	cfg := tinyTimmExecutableConfig()
	src := tinyTimmVisionSource(cfg)
	prefix := "vpm.blocks.0"
	qkv := src[prefix+".attn.qkv.weight"]
	qkv.data = []float32{1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1}
	src[prefix+".attn.qkv.weight"] = qkv
	proj := src[prefix+".attn.proj.weight"]
	proj.data = []float32{1, 0, 0, 1}
	src[prefix+".attn.proj.weight"] = proj
	model, err := LoadTimmSigLIPVisionCPU(src, cfg)
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
		t.Fatalf("non-deterministic first=%v second=%v err=%v", first, second, err)
	}
}

func TestTimmSigLIPVisionCPURejectsMalformed(t *testing.T) {
	cfg := tinyTimmExecutableConfig()
	if _, err := LoadTimmSigLIPVisionCPU(nil, cfg); err == nil {
		t.Fatal("accepted nil source")
	}
	if _, err := LoadTimmSigLIPVisionCPUFromDir(t.TempDir(), cfg); err == nil {
		t.Fatal("accepted missing checkpoint")
	}
	wrong := cfg
	wrong.VisionEncoder = "other"
	if _, err := LoadTimmSigLIPVisionCPU(tinyTimmVisionSource(cfg), wrong); err == nil || !strings.Contains(err.Error(), "unsupported") {
		t.Fatalf("encoder error=%v", err)
	}
	bad := tinyTimmVisionSource(cfg)
	position := bad["vpm.pos_embed"]
	position.shape = []int{1, 3, 2}
	position.data = make([]float32, 6)
	bad["vpm.pos_embed"] = position
	if _, err := LoadTimmSigLIPVisionCPU(bad, cfg); err == nil || !strings.Contains(err.Error(), "not square") {
		t.Fatalf("position error=%v", err)
	}
	model, err := LoadTimmSigLIPVisionCPU(tinyTimmVisionSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.EncodeImage([]float32{float32(math.NaN()), 0, 0}, [4]int{1, 3, 1, 1}); err == nil || !strings.Contains(err.Error(), "non-finite") {
		t.Fatalf("non-finite error=%v", err)
	}
}
