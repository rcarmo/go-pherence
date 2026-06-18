package model

import (
	"encoding/binary"
	"fmt"

	"github.com/rcarmo/go-pherence/loader/gguf"
	"github.com/rcarmo/go-pherence/tensor"
)

// LoadGemma4MTPDrafterGGUF loads the compact Gemma4 assistant/MTP GGUF into
// the existing q-only drafter graph. The GGUF BF16 tensor shapes are [in, out]
// with contiguous output rows; this loader preserves raw BF16 rows for the
// runtime matmul path and also keeps F32 mirrors for validation/fallback tests.
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
	validateShape := func(t gguf.TensorInfo, want ...int) error {
		if len(t.Shape) != len(want) {
			return fmt.Errorf("tensor %s shape=%v, want %v for Gemma4 MTP drafter graph", t.Name, t.Shape, want)
		}
		for i, dim := range want {
			if dim <= 0 || t.Shape[i] != uint64(dim) {
				return fmt.Errorf("tensor %s shape=%v, want %v for Gemma4 MTP drafter graph", t.Name, t.Shape, want)
			}
		}
		return nil
	}
	loadRowsBF16 := func(name string, inDim, outDim int) ([]float32, []uint16, error) {
		t, ok := g.TensorByName(name)
		if !ok {
			return nil, nil, fmt.Errorf("tensor %q not found", name)
		}
		if t.QType != gguf.QuantBF16 {
			return nil, nil, fmt.Errorf("tensor %s type=%s, want BF16 for Gemma4 MTP drafter graph", name, t.QType)
		}
		if err := validateShape(t, inDim, outDim); err != nil {
			return nil, nil, err
		}
		raw, err := g.Raw(t)
		if err != nil {
			return nil, nil, err
		}
		wantElems := inDim * outDim
		if len(raw) < wantElems*2 {
			return nil, nil, fmt.Errorf("tensor %s raw BF16 bytes=%d, want %d", name, len(raw), wantElems*2)
		}
		bf16 := make([]uint16, wantElems)
		for i := range bf16 {
			bf16[i] = binary.LittleEndian.Uint16(raw[i*2:])
		}
		m, err := g.MatrixFromTensor(t)
		if err != nil {
			return nil, nil, err
		}
		if m.InDim != inDim || m.OutDim != outDim {
			return nil, nil, fmt.Errorf("tensor %s dims out/in=%d/%d, want %d/%d", name, m.OutDim, m.InDim, outDim, inDim)
		}
		f32 := make([]float32, outDim*inDim)
		for row := 0; row < outDim; row++ {
			if err := m.DequantRowTo(f32[row*inDim:(row+1)*inDim], row); err != nil {
				return nil, nil, err
			}
		}
		return f32, bf16, nil
	}
	loadTensor := func(name string, shape []int) (*tensor.Tensor, error) {
		t, ok := g.TensorByName(name)
		if !ok {
			return nil, fmt.Errorf("tensor %q not found", name)
		}
		if t.QType != gguf.QuantF32 {
			return nil, fmt.Errorf("tensor %s type=%s, want F32 for Gemma4 MTP drafter graph", name, t.QType)
		}
		if err := validateShape(t, shape...); err != nil {
			return nil, err
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
	var fullRoPEFactors []float32
	if t, ok := g.TensorByName("rope_freqs.weight"); ok {
		if t.QType != gguf.QuantF32 {
			return nil, fmt.Errorf("rope_freqs.weight type=%s, want F32 for Gemma4 MTP drafter graph", t.QType)
		}
		fullRoPEFactors, err = g.DequantF32(t)
		if err != nil {
			return nil, fmt.Errorf("load rope_freqs.weight: %w", err)
		}
		if want := cfg.GlobalHeadDim / 2; want > 0 && len(fullRoPEFactors) != want {
			return nil, fmt.Errorf("rope_freqs.weight len=%d, want assistant global_head_dim/2=%d", len(fullRoPEFactors), want)
		}
	}
	d := &Gemma4MTPDrafter{Config: cfg, BackboneHiddenSize: backboneHidden, Layers: make([]Gemma4MTPDrafterLayer, cfg.NumLayers)}
	d.precomputeGemma4RoPEWithFullFactors(fullRoPEFactors)
	if emb, embBF16, err := loadRowsBF16("token_embd.weight", cfg.HiddenSize, cfg.VocabSize); err != nil {
		return nil, err
	} else {
		d.EmbedTokens = tensor.FromFloat32(emb, []int{cfg.VocabSize, cfg.HiddenSize})
		d.EmbedTokensBF16 = embBF16
	}
	if d.Norm, err = loadTensor("output_norm.weight", []int{cfg.HiddenSize}); err != nil {
		return nil, err
	}
	preWidth := 2 * backboneHidden
	if d.PreProjection, d.PreProjectionBF16, err = loadRowsBF16("nextn.pre_projection.weight", preWidth, cfg.HiddenSize); err != nil {
		return nil, err
	}
	if d.PostProjection, d.PostProjectionBF16, err = loadRowsBF16("nextn.post_projection.weight", cfg.HiddenSize, backboneHidden); err != nil {
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
		if layer.QW, layer.QWBF16, err = loadRowsBF16(p+"attn_q.weight", cfg.HiddenSize, qDim); err != nil {
			return nil, err
		}
		if layer.OW, layer.OWBF16, err = loadRowsBF16(p+"attn_output.weight", qDim, cfg.HiddenSize); err != nil {
			return nil, err
		}
		if layer.GateW, layer.GateWBF16, err = loadRowsBF16(p+"ffn_gate.weight", cfg.HiddenSize, cfg.Intermediate); err != nil {
			return nil, err
		}
		if layer.UpW, layer.UpWBF16, err = loadRowsBF16(p+"ffn_up.weight", cfg.HiddenSize, cfg.Intermediate); err != nil {
			return nil, err
		}
		if layer.DownW, layer.DownWBF16, err = loadRowsBF16(p+"ffn_down.weight", cfg.Intermediate, cfg.HiddenSize); err != nil {
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
	if nextn, err := u("nextn_predict_layers"); err == nil && nextn != layers {
		return LlamaConfig{}, 0, fmt.Errorf("%s nextn_predict_layers=%d, want block_count=%d", p, nextn, layers)
	}
	if shared, err := u("attention.shared_kv_layers"); err == nil && shared != layers {
		return LlamaConfig{}, 0, fmt.Errorf("%s attention.shared_kv_layers=%d, want block_count=%d for q-only assistant", p, shared, layers)
	}
	hidden, err := u("embedding_length")
	if err != nil {
		return LlamaConfig{}, 0, err
	}
	backbone, err := u("embedding_length_out")
	if err != nil {
		return LlamaConfig{}, 0, err
	}
	if backbone == hidden {
		return LlamaConfig{}, 0, fmt.Errorf("%s embedding_length_out=%d must differ from assistant embedding_length", p, backbone)
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
	if v, err := u("attention.value_length_swa"); err == nil && v != headDim {
		return LlamaConfig{}, 0, fmt.Errorf("%s attention.value_length_swa=%d, want key_length_swa=%d", p, v, headDim)
	}
	if v, err := u("attention.value_length"); err == nil && v != globalHeadDim {
		return LlamaConfig{}, 0, fmt.Errorf("%s attention.value_length=%d, want key_length=%d", p, v, globalHeadDim)
	}
	if v, err := u("rope.dimension_count_swa"); err == nil && v != headDim {
		return LlamaConfig{}, 0, fmt.Errorf("%s rope.dimension_count_swa=%d, want key_length_swa=%d", p, v, headDim)
	}
	if v, err := u("rope.dimension_count"); err == nil && v != globalHeadDim {
		return LlamaConfig{}, 0, fmt.Errorf("%s rope.dimension_count=%d, want key_length=%d", p, v, globalHeadDim)
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
	layerTypes := gemma4GGUFLayerTypesForKey(g, p+".attention.sliding_window_pattern", layers)
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
