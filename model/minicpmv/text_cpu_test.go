package minicpmv

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/config"
)

type fakeTextTensorSource map[string]struct {
	data  []float32
	shape []int
}

func (s fakeTextTensorSource) GetFloat32(name string) ([]float32, []int, error) {
	t, ok := s[name]
	if !ok {
		return nil, nil, fmt.Errorf("missing %s", name)
	}
	return append([]float32(nil), t.data...), append([]int(nil), t.shape...), nil
}

func boolPointer(v bool) *bool { return &v }

func tinyTextConfig(family string) config.MiniCPMVConfig {
	cfg := config.MiniCPMVConfig{
		Architectures: []string{"MiniCPMVForCausalLM"}, ModelType: "minicpmv",
		HiddenSize: 4, NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1,
		HeadDim: 2, IntermediateSize: 4, VocabSize: 4, MaxPositionEmbeds: 8,
		RMSNormEps: 1e-6, RopeTheta: 10000, HiddenAct: "silu", AttentionBias: boolPointer(false),
		NumQuery: 1,
	}
	switch family {
	case "minicpm":
		cfg.ScaleEmb, cfg.ScaleDepth, cfg.DimModelBase = 2, 2, 2
	case "qwen2":
		cfg.TextConfig = &config.MiniCPMVTextConfig{
			ModelType: "qwen2", HiddenSize: 4, NumHiddenLayers: 1, NumAttentionHeads: 2,
			NumKeyValueHeads: 1, HeadDim: 2, IntermediateSize: 4, VocabSize: 4,
			MaxPositionEmbeds: 8, RMSNormEps: 1e-6, RopeTheta: 10000, HiddenAct: "silu",
			AttentionBias: boolPointer(true), TieWordEmbeddings: boolPointer(false),
		}
	case "mistral":
		cfg.ModelType = "omnilmm"
		cfg.TextConfig = &config.MiniCPMVTextConfig{
			ModelType: "mistral", HiddenSize: 4, NumHiddenLayers: 1, NumAttentionHeads: 2,
			NumKeyValueHeads: 1, HeadDim: 2, IntermediateSize: 4, VocabSize: 4,
			MaxPositionEmbeds: 8, RMSNormEps: 1e-6, RopeTheta: 10000, HiddenAct: "silu",
			AttentionBias: boolPointer(false), TieWordEmbeddings: boolPointer(true),
		}
		cfg.TieWordEmbeddings = true
	}
	return cfg
}

func tinyTextSource(cfg config.MiniCPMVConfig) fakeTextTensorSource {
	s := cfg.MiniCPMVSummary()
	modelPrefix, headName := "llm.model", "llm.lm_head.weight"
	if s.TextModelType == "mistral" {
		modelPrefix, headName = "model", "lm_head.weight"
	}
	src := fakeTextTensorSource{}
	add := func(name string, shape []int, data []float32) {
		src[name] = struct {
			data  []float32
			shape []int
		}{data: data, shape: shape}
	}
	identity := func(rows, cols int) []float32 {
		out := make([]float32, rows*cols)
		for i := 0; i < min(rows, cols); i++ {
			out[i*cols+i] = 1
		}
		return out
	}
	embed := []float32{
		1, 2, 3, 4,
		4, 3, 2, 1,
		1, -1, 2, -2,
		2, 0, 0, 1,
	}
	add(modelPrefix+".embed_tokens.weight", []int{s.VocabSize, s.HiddenSize}, embed)
	add(modelPrefix+".norm.weight", []int{s.HiddenSize}, []float32{1, 1, 1, 1})
	if !cfg.TieWordEmbeddings && (cfg.TextConfig == nil || cfg.TextConfig.TieWordEmbeddings == nil || !*cfg.TextConfig.TieWordEmbeddings) {
		head := []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}
		add(headName, []int{s.VocabSize, s.HiddenSize}, head)
	}
	prefix := modelPrefix + ".layers.0"
	add(prefix+".input_layernorm.weight", []int{s.HiddenSize}, []float32{1, 1, 1, 1})
	add(prefix+".post_attention_layernorm.weight", []int{s.HiddenSize}, []float32{1, 1, 1, 1})
	queryWidth := s.Heads * s.HeadDim
	kvWidth := s.KVHeads * s.HeadDim
	add(prefix+".self_attn.q_proj.weight", []int{queryWidth, s.HiddenSize}, identity(queryWidth, s.HiddenSize))
	add(prefix+".self_attn.k_proj.weight", []int{kvWidth, s.HiddenSize}, make([]float32, kvWidth*s.HiddenSize))
	add(prefix+".self_attn.v_proj.weight", []int{kvWidth, s.HiddenSize}, make([]float32, kvWidth*s.HiddenSize))
	add(prefix+".self_attn.o_proj.weight", []int{s.HiddenSize, queryWidth}, make([]float32, s.HiddenSize*queryWidth))
	if cfg.TextConfig != nil && cfg.TextConfig.AttentionBias != nil && *cfg.TextConfig.AttentionBias {
		add(prefix+".self_attn.q_proj.bias", []int{queryWidth}, make([]float32, queryWidth))
		add(prefix+".self_attn.k_proj.bias", []int{kvWidth}, make([]float32, kvWidth))
		add(prefix+".self_attn.v_proj.bias", []int{kvWidth}, make([]float32, kvWidth))
	}
	for _, name := range []string{"gate_proj", "up_proj", "down_proj"} {
		add(prefix+".mlp."+name+".weight", []int{s.IntermediateSize, s.HiddenSize}, make([]float32, s.IntermediateSize*s.HiddenSize))
	}
	return src
}

func closeTextSlice(a, b []float32, tolerance float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > tolerance {
			return false
		}
	}
	return true
}

func TestTextCPUOneTokenFamiliesAndScaling(t *testing.T) {
	for _, family := range []string{"minicpm", "qwen2", "mistral"} {
		t.Run(family, func(t *testing.T) {
			cfg := tinyTextConfig(family)
			src := tinyTextSource(cfg)
			model, err := LoadTextCPU(src, cfg)
			if err != nil {
				t.Fatal(err)
			}
			// The model must own all decoded weights.
			embedName := model.cfg.modelPrefix + ".embed_tokens.weight"
			src[embedName].data[0] = 99
			result, next, err := model.ForwardToken(0, model.NewState())
			if err != nil {
				t.Fatal(err)
			}
			input := []float32{1, 2, 3, 4}
			logitScale := float32(1)
			if family == "minicpm" {
				input = []float32{2, 4, 6, 8}
				logitScale = .5
			}
			var sum float64
			for _, x := range input {
				sum += float64(x * x)
			}
			inv := float32(1 / math.Sqrt(sum/4+1e-6))
			wantHidden := make([]float32, 4)
			for i := range wantHidden {
				wantHidden[i] = input[i] * inv
			}
			if !closeTextSlice(result.Hidden, wantHidden, 2e-6) {
				t.Fatalf("hidden=%v want=%v", result.Hidden, wantHidden)
			}
			wantLogits := make([]float32, 4)
			if family == "mistral" {
				embed := tinyTextSource(cfg)["model.embed_tokens.weight"].data
				for row := range wantLogits {
					for col := 0; col < 4; col++ {
						wantLogits[row] += embed[row*4+col] * wantHidden[col]
					}
				}
			} else {
				for i := range wantLogits {
					wantLogits[i] = wantHidden[i] * logitScale
				}
			}
			if !closeTextSlice(result.Logits, wantLogits, 2e-6) || next.Pos != 1 || len(next.K) != 1 || len(next.K[0]) != 2 {
				t.Fatalf("logits/state=%v %+v", result.Logits, next)
			}
		})
	}
}

func TestTextCPUGenerateFromEmbeddingsAndRuntimeInterfaces(t *testing.T) {
	cfg := tinyTextConfig("qwen2")
	model, err := LoadTextCPU(tinyTextSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	embeddings := []float32{1, 2, 3, 4}
	generated, err := model.GenerateFromEmbeddings(embeddings, 1, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(generated) != 2 || generated[0] != 3 || generated[1] != 0 {
		t.Fatalf("generated=%v", generated)
	}
	rt, err := NewTextRuntimeInterfaces(model)
	if err != nil || rt.Text != model {
		t.Fatalf("runtime=%+v err=%v", rt, err)
	}
	if _, err := NewTextRuntimeInterfaces(nil); err == nil {
		t.Fatal("accepted nil text runtime")
	}
	for name, call := range map[string]func() error{
		"shape": func() error { _, err := model.GenerateFromEmbeddings([]float32{1}, 1, 4, 1); return err },
		"limit": func() error { _, err := model.GenerateFromEmbeddings(embeddings, 1, 4, 8); return err },
	} {
		t.Run(name, func(t *testing.T) {
			if err := call(); err == nil {
				t.Fatal("accepted malformed generation request")
			}
		})
	}
}

func TestTextCPUStateOwnershipAndDeterminism(t *testing.T) {
	cfg := tinyTextConfig("qwen2")
	src := tinyTextSource(cfg)
	prefix := "llm.model.layers.0.self_attn."
	k := src[prefix+"k_proj.weight"]
	copy(k.data, []float32{1, 0, 0, 0, 0, 1, 0, 0})
	src[prefix+"k_proj.weight"] = k
	v := src[prefix+"v_proj.weight"]
	copy(v.data, []float32{1, 0, 0, 0, 1, 0, 0, 0})
	src[prefix+"v_proj.weight"] = v
	o := src[prefix+"o_proj.weight"]
	copy(o.data, []float32{1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0})
	src[prefix+"o_proj.weight"] = o
	model, err := LoadTextCPU(src, cfg)
	if err != nil {
		t.Fatal(err)
	}
	initial := model.NewState()
	first, state1, err := model.ForwardToken(0, initial)
	if err != nil {
		t.Fatal(err)
	}
	second, state2, err := model.ForwardToken(0, initial)
	if err != nil {
		t.Fatal(err)
	}
	if !closeTextSlice(first.Hidden, second.Hidden, 0) || !closeTextSlice(first.Logits, second.Logits, 0) || initial.Pos != 0 || len(initial.K[0]) != 0 {
		t.Fatalf("non-deterministic result or mutated input state")
	}
	state2.K[0][0] = 99
	if state1.K[0][0] == 99 {
		t.Fatal("returned states alias")
	}
	first.Hidden[0] = 99
	third, _, err := model.ForwardToken(0, initial)
	if err != nil || third.Hidden[0] == 99 {
		t.Fatalf("returned hidden aliases model scratch: hidden=%v err=%v", third.Hidden, err)
	}
	continued, state3, err := model.ForwardToken(1, state1)
	if err != nil || state3.Pos != 2 || len(state3.K[0]) != 4 || len(continued.Logits) != 4 {
		t.Fatalf("continuation result/state/err=%+v %+v %v", continued, state3, err)
	}
}

func TestTextCPUForwardEmbeddingAndRejectsMalformedInput(t *testing.T) {
	cfg := tinyTextConfig("qwen2")
	model, err := LoadTextCPU(tinyTextSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	input := []float32{1, 2, 3, 4}
	result, state, err := model.ForwardEmbedding(input, model.NewState())
	if err != nil || len(result.Hidden) != 4 || len(result.Logits) != 4 || state.Pos != 1 {
		t.Fatalf("result/state/err=%+v %+v %v", result, state, err)
	}
	input[0] = 99
	if result.Hidden[0] == 99 {
		t.Fatal("result aliases caller embedding")
	}
	for name, embedding := range map[string][]float32{
		"short": {1},
		"nan":   {1, 2, float32(math.NaN()), 4},
		"inf":   {1, 2, 3, float32(math.Inf(1))},
	} {
		t.Run(name, func(t *testing.T) {
			if _, _, err := model.ForwardEmbedding(embedding, model.NewState()); err == nil {
				t.Fatal("accepted malformed embedding")
			}
		})
	}
	if _, _, err := model.ForwardToken(4, model.NewState()); err == nil {
		t.Fatal("accepted out-of-vocabulary token")
	}
	badState := model.NewState()
	badState.Pos = 1
	if _, _, err := model.ForwardToken(0, badState); err == nil || !strings.Contains(err.Error(), "KV lengths") {
		t.Fatalf("bad state error=%v", err)
	}
}

func TestLoadTextCPURejectsPolicyShapeMissingAndNonfinite(t *testing.T) {
	cfg := tinyTextConfig("qwen2")
	if _, err := LoadTextCPU(nil, cfg); err == nil {
		t.Fatal("accepted nil source")
	}
	if _, err := LoadTextCPUFromDir(t.TempDir(), cfg); err == nil {
		t.Fatal("accepted missing checkpoint")
	}
	badPolicy := tinyTextConfig("qwen2")
	badPolicy.TextConfig.RopeScaling = []byte(`{"type":"linear","factor":2}`)
	if _, err := LoadTextCPU(tinyTextSource(badPolicy), badPolicy); err == nil || !strings.Contains(err.Error(), "rope_scaling") {
		t.Fatalf("policy error=%v", err)
	}
	unsupported := tinyTextConfig("qwen2")
	unsupported.TextConfig.ModelType = "qwen3"
	if _, err := LoadTextCPU(tinyTextSource(unsupported), unsupported); err == nil || !strings.Contains(err.Error(), "model type") {
		t.Fatalf("family error=%v", err)
	}
	badShape := tinyTextSource(cfg)
	tensorValue := badShape["llm.lm_head.weight"]
	tensorValue.shape = []int{4, 3}
	badShape["llm.lm_head.weight"] = tensorValue
	if _, err := LoadTextCPU(badShape, cfg); err == nil || !strings.Contains(err.Error(), "shape=") {
		t.Fatalf("shape error=%v", err)
	}
	missing := tinyTextSource(cfg)
	delete(missing, "llm.model.norm.weight")
	if _, err := LoadTextCPU(missing, cfg); err == nil || !strings.Contains(err.Error(), "missing") {
		t.Fatalf("missing error=%v", err)
	}
	nonfinite := tinyTextSource(cfg)
	tensorValue = nonfinite["llm.model.embed_tokens.weight"]
	tensorValue.data[0] = float32(math.NaN())
	nonfinite["llm.model.embed_tokens.weight"] = tensorValue
	if _, err := LoadTextCPU(nonfinite, cfg); err == nil || !strings.Contains(err.Error(), "non-finite") {
		t.Fatalf("non-finite error=%v", err)
	}
}
