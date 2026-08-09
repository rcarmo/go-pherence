package model

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/loader/gguf"
	"github.com/rcarmo/go-pherence/model/common"
	"github.com/rcarmo/go-pherence/tensor"
)

// LoadGemma4GGUFAsLlama loads a Gemma4 GGUF checkpoint into the existing
// LlamaModel execution graph. The hot matrices stay in their original GGUF
// quantized form and execute through direct GGUF/QAT row-dot helpers; scalar
// norms, RoPE factors, and layer scales are validated as F32 graph tensors.
func LoadGemma4GGUFAsLlama(path string) (*LlamaModel, error) {
	g, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	defer g.Close()
	arch, _ := g.MetaString("general.architecture")
	if arch != "gemma4" {
		return nil, fmt.Errorf("GGUF architecture %q, want gemma4", arch)
	}
	cfg, err := gemma4GGUFConfig(g)
	if err != nil {
		return nil, err
	}
	validateShape := func(t gguf.TensorInfo, want ...int) error {
		if len(t.Shape) != len(want) {
			return fmt.Errorf("tensor %s shape=%v, want %v for Gemma4 graph", t.Name, t.Shape, want)
		}
		for i, dim := range want {
			if dim <= 0 || t.Shape[i] != uint64(dim) {
				return fmt.Errorf("tensor %s shape=%v, want %v for Gemma4 graph", t.Name, t.Shape, want)
			}
		}
		return nil
	}
	loadTyped := func(name string, wantType gguf.QuantType) ([]float32, error) {
		t, ok := g.TensorByName(name)
		if !ok {
			return nil, fmt.Errorf("tensor %q not found", name)
		}
		if t.QType != wantType {
			return nil, fmt.Errorf("tensor %s type=%s, want %s for Gemma4 graph", name, t.QType, wantType)
		}
		return g.DequantF32(t)
	}
	loadTensor := func(name string, shape []int) (*tensor.Tensor, error) {
		t, ok := g.TensorByName(name)
		if !ok {
			return nil, fmt.Errorf("tensor %q not found", name)
		}
		if t.QType != gguf.QuantF32 {
			return nil, fmt.Errorf("tensor %s type=%s, want F32 for Gemma4 graph", name, t.QType)
		}
		data, err := g.DequantF32(t)
		if err != nil {
			return nil, err
		}
		want := 1
		for _, d := range shape {
			if d <= 0 {
				return nil, fmt.Errorf("invalid shape for %s: %v", name, shape)
			}
			want *= d
		}
		if len(data) != want {
			return nil, fmt.Errorf("tensor %s len=%d, want %d for shape %v", name, len(data), want, shape)
		}
		return tensor.FromFloat32(data, shape), nil
	}
	loadQMatrix := func(name string) (*gguf.QuantMatrix, error) {
		t, ok := g.TensorByName(name)
		if !ok {
			return nil, fmt.Errorf("tensor %q not found", name)
		}
		return g.MatrixFromTensor(t)
	}
	loadProjection := func(name string) (*gguf.QuantMatrix, error) {
		matrix, err := loadQMatrix(name)
		if err != nil {
			return nil, err
		}
		if _, err := matrix.PrepareLlamaQ4_0x8(); err != nil {
			return nil, err
		}
		return matrix, nil
	}
	m := &LlamaModel{Config: cfg, Large: cfg.HiddenSize >= 3000, Quantized: true, SuppressTokens: ggufIntArray(g.Meta["tokenizer.ggml.suppress_tokens"])}
	if m.EmbedTokensGGUF, err = loadQMatrix("token_embd.weight"); err != nil {
		return nil, err
	}
	if _, ok := g.TensorByName("output.weight"); ok {
		if m.LMHeadGGUF, err = loadQMatrix("output.weight"); err != nil {
			return nil, err
		}
	} else {
		// llama.cpp creates output.weight as optional and ties it to tok_embd
		// only when the GGUF omits a separate LM head.
		m.LMHeadGGUF = m.EmbedTokensGGUF
	}
	if m.Norm, err = loadTensor("output_norm.weight", []int{cfg.HiddenSize}); err != nil {
		return nil, err
	}
	m.Layers = make([]LlamaLayer, cfg.NumLayers)
	firstKVShared := cfg.NumLayers - cfg.NumKVSharedLayers
	for l := 0; l < cfg.NumLayers; l++ {
		p := fmt.Sprintf("blk.%d.", l)
		layer := &m.Layers[l]
		lt := "sliding_attention"
		if l < len(cfg.LayerTypes) {
			lt = cfg.LayerTypes[l]
		}
		if lt == "full_attention" && cfg.GlobalHeadDim > 0 {
			layer.HeadDimLocal = cfg.GlobalHeadDim
		} else {
			layer.HeadDimLocal = cfg.HeadDim
		}
		if l < firstKVShared || cfg.NumKVSharedLayers == 0 {
			layer.HasKV = true
		} else {
			layer.HasKV = false
			for src := 0; src < firstKVShared; src++ {
				if src < len(cfg.LayerTypes) && cfg.LayerTypes[src] == lt {
					layer.KVSourceLayer = src
				}
			}
		}
		if layer.InputNorm, err = loadTensor(p+"attn_norm.weight", []int{cfg.HiddenSize}); err != nil {
			return nil, err
		}
		if layer.PostNorm, err = loadTensor(p+"post_attention_norm.weight", []int{cfg.HiddenSize}); err != nil {
			return nil, err
		}
		if layer.PreFFNNorm, err = loadTensor(p+"ffn_norm.weight", []int{cfg.HiddenSize}); err != nil {
			return nil, err
		}
		if layer.PostFFNNorm, err = loadTensor(p+"post_ffw_norm.weight", []int{cfg.HiddenSize}); err != nil {
			return nil, err
		}
		if layer.QNorm, err = loadTensor(p+"attn_q_norm.weight", []int{layer.HeadDimLocal}); err != nil {
			return nil, err
		}
		if layer.HasKV {
			if layer.KNorm, err = loadTensor(p+"attn_k_norm.weight", []int{layer.HeadDimLocal}); err != nil {
				return nil, err
			}
		} else {
			// Shared-KV layers still need a K norm only if they materialize K, which they do not.
			if t, ok := g.TensorByName(p + "attn_k_norm.weight"); ok {
				if data, derr := g.DequantF32(t); derr == nil && len(data) == layer.HeadDimLocal {
					layer.KNorm = tensor.FromFloat32(data, []int{layer.HeadDimLocal})
				}
			}
		}
		if layer.QWGGUF, err = loadProjection(p + "attn_q.weight"); err != nil {
			return nil, err
		}
		if layer.HasKV {
			if layer.KWGGUF, err = loadProjection(p + "attn_k.weight"); err != nil {
				return nil, err
			}
			if layer.VWGGUF, err = loadProjection(p + "attn_v.weight"); err != nil {
				if _, ok := g.TensorByName(p + "attn_v.weight"); ok {
					return nil, err
				}
				// llama.cpp Gemma4 treats v_proj as optional and uses Kcur as Vcur
				// when it is absent (use_alternative_attention / K=V).
				layer.VWGGUF = layer.KWGGUF
				cfg.AttentionKEqV = true
				m.Config.AttentionKEqV = true
			}
		}
		if layer.OWGGUF, err = loadProjection(p + "attn_output.weight"); err != nil {
			return nil, err
		}
		if layer.GateWGGUF, err = loadProjection(p + "ffn_gate.weight"); err != nil {
			return nil, err
		}
		if layer.UpWGGUF, err = loadProjection(p + "ffn_up.weight"); err != nil {
			return nil, err
		}
		if layer.DownWGGUF, err = loadProjection(p + "ffn_down.weight"); err != nil {
			return nil, err
		}
		if t, ok := g.TensorByName(p + "layer_output_scale.weight"); ok {
			if t.QType != gguf.QuantF32 {
				return nil, fmt.Errorf("tensor %slayer_output_scale.weight type=%s, want F32 for Gemma4 graph", p, t.QType)
			}
			if gguf.TensorElements(t.Shape) != 1 {
				return nil, fmt.Errorf("tensor %slayer_output_scale.weight shape=%v, want scalar [1] for Gemma4 graph", p, t.Shape)
			}
			if sc, err := g.DequantF32(t); err == nil && len(sc) == 1 {
				layer.LayerScalar = sc[0]
			} else if err != nil {
				return nil, err
			} else {
				return nil, fmt.Errorf("tensor %slayer_output_scale.weight decoded len=%d, want 1", p, len(sc))
			}
		} else {
			layer.LayerScalar = 1
		}
		if cfg.HiddenPerLayer > 0 {
			if layer.PLIGateGGUF, err = loadProjection(p + "inp_gate.weight"); err != nil {
				return nil, err
			}
			if layer.PLIProjGGUF, err = loadProjection(p + "proj.weight"); err != nil {
				return nil, err
			}
			if layer.PLIPostNorm, err = loadTyped(p+"post_norm.weight", gguf.QuantF32); err != nil {
				return nil, err
			}
		}
	}
	totalPerLayerDim, ok := checkedProduct(cfg.NumLayers, cfg.HiddenPerLayer)
	if !ok || totalPerLayerDim <= 0 {
		return nil, fmt.Errorf("Gemma4 per-layer input dimension overflow layers=%d hiddenPerLayer=%d", cfg.NumLayers, cfg.HiddenPerLayer)
	}
	if t, ok := g.TensorByName("per_layer_model_proj.weight"); !ok {
		return nil, fmt.Errorf("tensor %q not found", "per_layer_model_proj.weight")
	} else if err := validateShape(t, cfg.HiddenSize, totalPerLayerDim); err != nil {
		return nil, err
	}
	if m.PerLayerModelProj, err = loadTyped("per_layer_model_proj.weight", gguf.QuantF16); err != nil {
		return nil, err
	}
	if t, ok := g.TensorByName("per_layer_proj_norm.weight"); !ok {
		return nil, fmt.Errorf("tensor %q not found", "per_layer_proj_norm.weight")
	} else if err := validateShape(t, cfg.HiddenPerLayer); err != nil {
		return nil, err
	}
	if m.PerLayerProjNorm, err = loadTyped("per_layer_proj_norm.weight", gguf.QuantF32); err != nil {
		return nil, err
	}
	if t, ok := g.TensorByName("per_layer_token_embd.weight"); !ok {
		return nil, fmt.Errorf("tensor %q not found", "per_layer_token_embd.weight")
	} else if err := validateShape(t, totalPerLayerDim, cfg.VocabPerLayer); err != nil {
		return nil, err
	}
	if m.EmbedPerLayerGGUF, err = loadQMatrix("per_layer_token_embd.weight"); err != nil {
		return nil, err
	}
	m.PerLayerInputScale = float32(1 / math.Sqrt(2))
	m.PerLayerProjScale = float32(1 / math.Sqrt(float64(cfg.HiddenSize)))
	m.EmbedPerLayerScale = float32(math.Sqrt(float64(cfg.HiddenPerLayer)))
	m.precomputeRoPE()
	var fullRoPEFactors []float32
	if t, ok := g.TensorByName("rope_freqs.weight"); ok {
		if t.QType != gguf.QuantF32 {
			return nil, fmt.Errorf("rope_freqs.weight type=%s, want F32 for Gemma4 graph", t.QType)
		}
		fullRoPEFactors, err = g.DequantF32(t)
		if err != nil {
			return nil, fmt.Errorf("load rope_freqs.weight: %w", err)
		}
		if want := cfg.GlobalHeadDim / 2; want > 0 && len(fullRoPEFactors) != want {
			return nil, fmt.Errorf("rope_freqs.weight len=%d, want global_head_dim/2=%d", len(fullRoPEFactors), want)
		}
	}
	m.precomputeGemma4RoPEWithFullFactors(fullRoPEFactors)
	return m, nil
}

func gemma4GGUFConfig(g *gguf.GGUF) (common.Config, error) {
	req := func(key string) (int, error) {
		v, ok := g.MetaUint32(key)
		if !ok {
			return 0, fmt.Errorf("missing metadata %s", key)
		}
		return int(v), nil
	}
	h, err := req("gemma4.embedding_length")
	if err != nil {
		return common.Config{}, err
	}
	layers, err := req("gemma4.block_count")
	if err != nil {
		return common.Config{}, err
	}
	heads, err := req("gemma4.attention.head_count")
	if err != nil {
		return common.Config{}, err
	}
	kvHeads, err := req("gemma4.attention.head_count_kv")
	if err != nil {
		return common.Config{}, err
	}
	inter, err := req("gemma4.feed_forward_length")
	if err != nil {
		return common.Config{}, err
	}
	ctx, err := req("gemma4.context_length")
	if err != nil {
		return common.Config{}, err
	}
	keyLen, err := req("gemma4.attention.key_length_swa")
	if err != nil {
		return common.Config{}, err
	}
	globalKeyLen, err := req("gemma4.attention.key_length")
	if err != nil {
		return common.Config{}, err
	}
	if v, err := req("gemma4.attention.value_length_swa"); err == nil && v != keyLen {
		return common.Config{}, fmt.Errorf("gemma4 value_length_swa=%d, want key_length_swa=%d", v, keyLen)
	}
	if v, err := req("gemma4.attention.value_length"); err == nil && v != globalKeyLen {
		return common.Config{}, fmt.Errorf("gemma4 value_length=%d, want key_length=%d", v, globalKeyLen)
	}
	if v, err := req("gemma4.rope.dimension_count_swa"); err == nil && v != keyLen {
		return common.Config{}, fmt.Errorf("gemma4 rope.dimension_count_swa=%d, want key_length_swa=%d", v, keyLen)
	}
	if v, err := req("gemma4.rope.dimension_count"); err == nil && v != globalKeyLen {
		return common.Config{}, fmt.Errorf("gemma4 rope.dimension_count=%d, want key_length=%d", v, globalKeyLen)
	}
	sliding, _ := req("gemma4.attention.sliding_window")
	shared, _ := req("gemma4.attention.shared_kv_layers")
	hpl, _ := req("gemma4.embedding_length_per_layer_input")
	vocab := 0
	if t, ok := g.TensorByName("token_embd.weight"); ok && len(t.Shape) >= 2 {
		vocab = int(t.Shape[1])
	}
	if vocab == 0 {
		return common.Config{}, fmt.Errorf("could not infer vocab size")
	}
	lt := gemma4GGUFLayerTypes(g, layers)
	eps := float64(1e-6)
	if v, ok := g.MetaFloat32("gemma4.attention.layer_norm_rms_epsilon"); ok {
		eps = float64(v)
	}
	ropeTheta := float64(1000000)
	if v, ok := g.MetaFloat32("gemma4.rope.freq_base"); ok {
		ropeTheta = float64(v)
	}
	bos := 2
	if v, ok := g.MetaUint32("tokenizer.ggml.bos_token_id"); ok {
		bos = int(v)
	}
	softcap := float32(0)
	if v, ok := g.MetaFloat32("gemma4.final_logit_softcapping"); ok {
		softcap = v
	}
	return common.Config{VocabSize: vocab, HiddenSize: h, Intermediate: inter, NumLayers: layers, NumHeads: heads, NumKVHeads: kvHeads, NumGlobalKVHeads: kvHeads, MaxSeqLen: ctx, RopeTheta: ropeTheta, RMSNormEps: eps, ModelType: "gemma4_text", HeadDim: keyLen, GlobalHeadDim: globalKeyLen, SlidingWindow: sliding, BOSTokenID: bos, LayerTypes: lt, NumKVSharedLayers: shared, HiddenPerLayer: hpl, VocabPerLayer: vocab, HiddenAct: "gelu_pytorch_tanh", AttentionKEqV: false, FinalLogitSoftcapping: float64(softcap)}, nil
}

func ggufIntArray(v any) []int {
	switch arr := v.(type) {
	case []any:
		out := make([]int, 0, len(arr))
		for _, x := range arr {
			switch n := x.(type) {
			case int:
				out = append(out, n)
			case int32:
				out = append(out, int(n))
			case uint32:
				out = append(out, int(n))
			case int64:
				out = append(out, int(n))
			case uint64:
				out = append(out, int(n))
			case float64:
				out = append(out, int(n))
			}
		}
		return out
	case []int32:
		out := make([]int, len(arr))
		for i, x := range arr {
			out[i] = int(x)
		}
		return out
	case []uint32:
		out := make([]int, len(arr))
		for i, x := range arr {
			out[i] = int(x)
		}
		return out
	}
	return nil
}

func gemma4GGUFLayerTypes(g *gguf.GGUF, n int) []string {
	return gemma4GGUFLayerTypesForKey(g, "gemma4.attention.sliding_window_pattern", n)
}

func gemma4GGUFLayerTypesForKey(g *gguf.GGUF, key string, n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = "sliding_attention"
	}
	arr, ok := g.Meta[key].([]any)
	if ok && len(arr) >= n {
		for i := 0; i < n; i++ {
			if b, ok := arr[i].(bool); ok && !b {
				out[i] = "full_attention"
			}
		}
		return out
	}
	if arrb, ok := g.Meta[key].([]bool); ok && len(arrb) >= n {
		for i := 0; i < n; i++ {
			if !arrb[i] {
				out[i] = "full_attention"
			}
		}
	}
	return out
}
