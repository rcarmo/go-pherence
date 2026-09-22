package minicpmv

import (
	"math"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/config"
)

func tinyResamplerConfig() config.MiniCPMVConfig {
	return config.MiniCPMVConfig{
		Architectures: []string{"MiniCPMOForCausalLM"}, ModelType: "minicpmo",
		HiddenSize: 4, NumHiddenLayers: 1, NumAttentionHeads: 1, NumKeyValueHeads: 1,
		IntermediateSize: 4, VocabSize: 4, NumQuery: 1,
		VisionConfig:    &config.MiniCPMVVisionConfig{ModelType: "siglip_vision_model", HiddenSize: 2, IntermediateSize: 3, NumHiddenLayers: 1, NumAttentionHeads: 1, ImageSize: 2, PatchSize: 1},
		ResamplerConfig: &config.MiniCPMVResamplerConfig{NumQuery: 1, NumHeads: 1, EmbedDim: 4, KVDim: 2},
	}
}

func tinyResamplerSource(cfg config.MiniCPMVConfig, withPosition bool) fakeTextTensorSource {
	s := cfg.MiniCPMVSummary()
	src := fakeTextTensorSource{}
	add := func(name string, shape []int, data []float32) {
		src[name] = struct {
			data  []float32
			shape []int
		}{data: data, shape: shape}
	}
	identity := func(dim int) []float32 {
		out := make([]float32, dim*dim)
		for i := 0; i < dim; i++ {
			out[i*dim+i] = 1
		}
		return out
	}
	add("resampler.query", []int{s.NumQuery, s.HiddenSize}, []float32{1, -1, 1, -1})
	if withPosition {
		add("resampler.pos_embed", []int{s.NumQuery, s.HiddenSize}, make([]float32, s.NumQuery*s.HiddenSize))
	}
	add("resampler.kv_proj.weight", []int{s.HiddenSize, s.VisionHiddenSize}, []float32{1, 0, 0, 1, 1, 0, 0, 1})
	packed := make([]float32, 3*s.HiddenSize*s.HiddenSize)
	copy(packed[:s.HiddenSize*s.HiddenSize], identity(s.HiddenSize))
	copy(packed[s.HiddenSize*s.HiddenSize:2*s.HiddenSize*s.HiddenSize], identity(s.HiddenSize))
	copy(packed[2*s.HiddenSize*s.HiddenSize:], identity(s.HiddenSize))
	add("resampler.attn.in_proj_weight", []int{3 * s.HiddenSize, s.HiddenSize}, packed)
	add("resampler.attn.in_proj_bias", []int{3 * s.HiddenSize}, make([]float32, 3*s.HiddenSize))
	add("resampler.attn.out_proj.weight", []int{s.HiddenSize, s.HiddenSize}, identity(s.HiddenSize))
	add("resampler.attn.out_proj.bias", []int{s.HiddenSize}, make([]float32, s.HiddenSize))
	for _, name := range []string{"ln_q", "ln_kv", "ln_post"} {
		add("resampler."+name+".weight", []int{s.HiddenSize}, []float32{1, 1, 1, 1})
		add("resampler."+name+".bias", []int{s.HiddenSize}, make([]float32, s.HiddenSize))
	}
	add("resampler.proj", []int{s.HiddenSize, s.HiddenSize}, identity(s.HiddenSize))
	return src
}

func TestResamplerCPUOneTokenSynthetic(t *testing.T) {
	cfg := tinyResamplerConfig()
	for _, withPosition := range []bool{false, true} {
		t.Run(map[bool]string{false: "dynamic-position", true: "stored-position"}[withPosition], func(t *testing.T) {
			src := tinyResamplerSource(cfg, withPosition)
			if withPosition {
				position := src["resampler.pos_embed"]
				position.data = []float32{2, -2, 2, -2}
				src["resampler.pos_embed"] = position
			}
			model, err := LoadResamplerCPU(src, cfg)
			if err != nil {
				t.Fatal(err)
			}
			src["resampler.query"].data[0] = 99
			out, err := model.Resample([]float32{1, 0}, 1, 2)
			if err != nil {
				t.Fatal(err)
			}
			if len(out) != 4 {
				t.Fatalf("output=%v", out)
			}
			// One key means attention probability one. KV projection yields
			// [1,0,1,0], and post LayerNorm preserves its alternating sign.
			want := []float32{1, -1, 1, -1}
			for i := range want {
				if math.Abs(float64(out[i]-want[i])) > 2e-5 {
					t.Fatalf("output[%d]=%g want %g; all=%v", i, out[i], want[i], out)
				}
			}
			out[0] = 99
			again, err := model.Resample([]float32{1, 0}, 1, 2)
			if err != nil || again[0] == 99 {
				t.Fatalf("owned rerun=%v err=%v", again, err)
			}
		})
	}
}

func TestResamplerCPUFourTokenDeterminism(t *testing.T) {
	cfg := tinyResamplerConfig()
	model, err := LoadResamplerCPU(tinyResamplerSource(cfg, false), cfg)
	if err != nil {
		t.Fatal(err)
	}
	vision := []float32{1, 0, 0, 1, 2, 1, -1, 2}
	first, err := model.Resample(vision, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	second, err := model.Resample(vision, 4, 2)
	if err != nil || !closeTextSlice(first, second, 0) {
		t.Fatalf("non-deterministic first=%v second=%v err=%v", first, second, err)
	}
}

func TestResamplerCPULegacyRequiresStoredQueryPositions(t *testing.T) {
	cfg := tinyResamplerConfig()
	cfg.ModelType = "minicpmv"
	cfg.VisionConfig.ModelType = "eva02"
	if _, err := LoadResamplerCPU(tinyResamplerSource(cfg, false), cfg); err == nil || !strings.Contains(err.Error(), "pos_embed") {
		t.Fatalf("missing legacy position error=%v", err)
	}
	if _, err := LoadResamplerCPU(tinyResamplerSource(cfg, true), cfg); err != nil {
		t.Fatalf("stored legacy position rejected: %v", err)
	}
}

func TestResamplerCPURejectsMalformedInputs(t *testing.T) {
	cfg := tinyResamplerConfig()
	if _, err := LoadResamplerCPU(nil, cfg); err == nil {
		t.Fatal("accepted nil source")
	}
	if _, err := LoadResamplerCPUFromDir(t.TempDir(), cfg); err == nil {
		t.Fatal("accepted missing checkpoint")
	}
	badShape := tinyResamplerSource(cfg, false)
	tensor := badShape["resampler.attn.in_proj_weight"]
	tensor.shape = []int{4, 12}
	badShape["resampler.attn.in_proj_weight"] = tensor
	if _, err := LoadResamplerCPU(badShape, cfg); err == nil || !strings.Contains(err.Error(), "shape=") {
		t.Fatalf("shape error=%v", err)
	}
	model, err := LoadResamplerCPU(tinyResamplerSource(cfg, false), cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		values []float32
		tokens int
		hidden int
	}{
		{make([]float32, 6), 3, 2},
		{make([]float32, 8), 4, 3},
		{[]float32{1}, 1, 2},
		{[]float32{float32(math.NaN()), 0}, 1, 2},
	} {
		if _, err := model.Resample(tc.values, tc.tokens, tc.hidden); err == nil {
			t.Fatalf("accepted malformed input %+v", tc)
		}
	}
}

func TestResampler2DPositions(t *testing.T) {
	positions := resampler2DPositions(2, 2, 4)
	if len(positions) != 16 || positions[0] != 0 || positions[1] != 1 || positions[2] != 0 || positions[3] != 1 {
		t.Fatalf("positions=%v", positions)
	}
	if got := positions[4]; math.Abs(float64(got)-math.Sin(1)) > 1e-6 {
		t.Fatalf("x-axis sin=%g", got)
	}
	if resampler2DPositions(1, 1, 6) != nil {
		t.Fatal("accepted embed dim not divisible by four")
	}
}
