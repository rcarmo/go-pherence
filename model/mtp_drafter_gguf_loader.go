package model

import (
	"fmt"

	"github.com/rcarmo/go-pherence/loader/gguf"
	"github.com/rcarmo/go-pherence/tensor"
)

// LoadGemma4MTPDrafterGGUF loads the compact Gemma4 assistant/MTP GGUF into
// the existing q-only drafter graph. The GGUF tensor shapes are [in, out], with
// contiguous output rows; this loader dequantizes the small BF16 projection and
// layer matrices to the dense [out, in] layout consumed by gemvNT.
func LoadGemma4MTPDrafterGGUF(path string) (*Gemma4MTPDrafter, error) {
	g, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	defer g.Close()
	arch, _ := g.MetaString("general.architecture")
	if arch != "gemma4-assistant" {
		return nil, fmt.Errorf("GGUF architecture %q, want gemma4-assistant", arch)
	}
	cfg, backboneHidden, err := gemma4MTPDrafterGGUFConfig(g)
	if err != nil {
		return nil, err
	}
	loadRows := func(name string, inDim, outDim int) ([]float32, error) {
		t, ok := g.TensorByName(name)
		if !ok {
			return nil, fmt.Errorf("tensor %q not found", name)
		}
		m, err := g.MatrixFromTensor(t)
		if err != nil {
			return nil, err
		}
		if m.InDim != inDim || m.OutDim != outDim {
			return nil, fmt.Errorf("tensor %s dims out/in=%d/%d, want %d/%d", name, m.OutDim, m.InDim, outDim, inDim)
		}
		out := make([]float32, outDim*inDim)
		for row := 0; row < outDim; row++ {
			if err := m.DequantRowTo(out[row*inDim:(row+1)*inDim], row); err != nil {
				return nil, err
			}
		}
		return out, nil
	}
	loadTensor := func(name string, shape []int) (*tensor.Tensor, error) {
		t, ok := g.TensorByName(name)
		if !ok {
			return nil, fmt.Errorf("tensor %q not found", name)
		}
		data, err := g.DequantF32(t)
		if err != nil {
			return nil, err
		}
		want := 1
		for _, d := range shape {
			want *= d
		}
		if len(data) != want {
			return nil, fmt.Errorf("tensor %s len=%d, want %d for shape %v", name, len(data), want, shape)
		}
		return tensor.FromFloat32(data, shape), nil
	}
	d := &Gemma4MTPDrafter{Config: cfg, BackboneHiddenSize: backboneHidden, Layers: make([]Gemma4MTPDrafterLayer, cfg.NumLayers)}
	if emb, err := loadRows("token_embd.weight", cfg.HiddenSize, cfg.VocabSize); err != nil {
		return nil, err
	} else {
		d.EmbedTokens = tensor.FromFloat32(emb, []int{cfg.VocabSize, cfg.HiddenSize})
	}
	if d.Norm, err = loadTensor("output_norm.weight", []int{cfg.HiddenSize}); err != nil {
		return nil, err
	}
	preWidth := 2 * backboneHidden
	if d.PreProjection, err = loadRows("nextn.pre_projection.weight", preWidth, cfg.HiddenSize); err != nil {
		return nil, err
	}
	if d.PostProjection, err = loadRows("nextn.post_projection.weight", cfg.HiddenSize, backboneHidden); err != nil {
		return nil, err
	}
	for l := 0; l < cfg.NumLayers; l++ {
		p := fmt.Sprintf("blk.%d.", l)
		layer := &d.Layers[l]
		layer.KVSourceLayer = -1
		if l < len(cfg.LayerTypes) && cfg.LayerTypes[l] == "full_attention" {
			layer.HeadDimLocal = cfg.GlobalHeadDim
		} else {
			layer.HeadDimLocal = cfg.HeadDim
		}
		qDim := cfg.NumHeads * layer.HeadDimLocal
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
		if s, err := loadTensor(p+"layer_output_scale.weight", []int{1}); err != nil {
			return nil, err
		} else {
			layer.LayerScalar = s.Data()[0]
		}
		if layer.QW, err = loadRows(p+"attn_q.weight", cfg.HiddenSize, qDim); err != nil {
			return nil, err
		}
		if layer.OW, err = loadRows(p+"attn_output.weight", qDim, cfg.HiddenSize); err != nil {
			return nil, err
		}
		if layer.GateW, err = loadRows(p+"ffn_gate.weight", cfg.HiddenSize, cfg.Intermediate); err != nil {
			return nil, err
		}
		if layer.UpW, err = loadRows(p+"ffn_up.weight", cfg.HiddenSize, cfg.Intermediate); err != nil {
			return nil, err
		}
		if layer.DownW, err = loadRows(p+"ffn_down.weight", cfg.Intermediate, cfg.HiddenSize); err != nil {
			return nil, err
		}
	}
	return d, nil
}

func gemma4MTPDrafterGGUFConfig(g *gguf.GGUF) (LlamaConfig, int, error) {
	p := "gemma4-assistant"
	u := func(key string) (int, error) {
		v, ok := g.MetaUint32(p + "." + key)
		if !ok {
			return 0, fmt.Errorf("missing metadata %s.%s", p, key)
		}
		return int(v), nil
	}
	layers, err := u("block_count")
	if err != nil {
		return LlamaConfig{}, 0, err
	}
	hidden, err := u("embedding_length")
	if err != nil {
		return LlamaConfig{}, 0, err
	}
	backbone, err := u("embedding_length_out")
	if err != nil {
		return LlamaConfig{}, 0, err
	}
	inter, err := u("feed_forward_length")
	if err != nil {
		return LlamaConfig{}, 0, err
	}
	ctx, err := u("context_length")
	if err != nil {
		return LlamaConfig{}, 0, err
	}
	heads, err := u("attention.head_count")
	if err != nil {
		return LlamaConfig{}, 0, err
	}
	kvHeads, err := u("attention.head_count_kv")
	if err != nil {
		return LlamaConfig{}, 0, err
	}
	headDim, err := u("attention.key_length_swa")
	if err != nil {
		return LlamaConfig{}, 0, err
	}
	globalHeadDim, err := u("attention.key_length")
	if err != nil {
		return LlamaConfig{}, 0, err
	}
	sw, err := u("attention.sliding_window")
	if err != nil {
		return LlamaConfig{}, 0, err
	}
	eps, ok := g.MetaFloat32(p + ".attention.layer_norm_rms_epsilon")
	if !ok {
		return LlamaConfig{}, 0, fmt.Errorf("missing metadata %s.attention.layer_norm_rms_epsilon", p)
	}
	vocab := 0
	if raw, ok := g.Meta["tokenizer.ggml.tokens"]; ok {
		if arr, ok := raw.([]any); ok {
			vocab = len(arr)
		}
	}
	if vocab == 0 {
		return LlamaConfig{}, 0, fmt.Errorf("missing tokenizer vocabulary")
	}
	layerTypes := make([]string, layers)
	for i := range layerTypes {
		layerTypes[i] = "sliding_attention"
	}
	if layers > 0 {
		layerTypes[layers-1] = "full_attention"
	}
	bos := 2
	if v, ok := g.MetaUint32("tokenizer.ggml.bos_token_id"); ok {
		bos = int(v)
	}
	return LlamaConfig{
		VocabSize: vocab, HiddenSize: hidden, Intermediate: inter, NumLayers: layers,
		NumHeads: heads, NumKVHeads: kvHeads, NumGlobalKVHeads: kvHeads,
		MaxSeqLen: ctx, RopeTheta: 1000000, RMSNormEps: float64(eps), ModelType: "gemma4_text",
		Architectures: []string{"Gemma4ForCausalLM"}, TieEmbeddings: true,
		HeadDim: headDim, GlobalHeadDim: globalHeadDim, SlidingWindow: sw,
		BOSTokenID: bos, LayerTypes: layerTypes, NumKVSharedLayers: layers,
	}, backbone, nil
}
