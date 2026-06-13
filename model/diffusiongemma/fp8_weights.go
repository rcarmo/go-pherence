package diffusiongemma

import (
	"fmt"
	"path/filepath"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// FP8LayerWeights holds per-layer FP8 weight data + scales for GPU upload.
type FP8LayerWeights struct {
	Layer int

	// Attention projections: FP8 weight bytes + F32 per-channel scale
	QWeight []byte
	QScale  []float32
	QShape  [2]int // [outDim, inDim]

	KWeight []byte
	KScale  []float32
	KShape  [2]int

	VWeight []byte
	VScale  []float32
	VShape  [2]int

	OWeight []byte
	OScale  []float32
	OShape  [2]int

	// MLP projections
	GateWeight []byte
	GateScale  []float32
	GateShape  [2]int

	UpWeight []byte
	UpScale  []float32
	UpShape  [2]int

	DownWeight []byte
	DownScale  []float32
	DownShape  [2]int

	// Norms (full precision F32/BF16)
	InputNorm    []float32
	PostAttnNorm []float32
	PreFFNNorm   []float32
	PreFFNNorm2  []float32
	PostFFNNorm  []float32
	PostFFNNorm1 []float32
	PostFFNNorm2 []float32

	// Attention norms
	QNorm []float32
	KNorm []float32

	// Router (full precision, not quantized)
	RouterScale          []float32
	RouterProj           []float32 // NOT FP8
	RouterProjShape      [2]int
	RouterPerExpertScale []float32

	// Layer scalar
	LayerScalar float32
}

// FP8TextWeights holds all weights for FP8 DiffusionGemma inference.
type FP8TextWeights struct {
	EmbedTokens       []float32 // Full precision embeddings
	EmbedShape        [2]int
	FinalNorm         []float32
	SelfCondPreNorm   []float32
	SelfCondGate      []float32
	SelfCondGateShape [2]int
	SelfCondUp        []float32
	SelfCondUpShape   [2]int
	SelfCondDown      []float32
	SelfCondDownShape [2]int
	Layers            []FP8LayerWeights
	shards            *safetensors.ShardedFile
}

// OpenFP8TextWeights opens an FP8-dynamic DiffusionGemma checkpoint.
func OpenFP8TextWeights(modelDir string, shape Shape) (*FP8TextWeights, error) {
	indexPath := filepath.Join(modelDir, "model.safetensors.index.json")
	shards, err := safetensors.OpenSharded(indexPath)
	if err != nil {
		return nil, fmt.Errorf("FP8 open shards: %w", err)
	}

	out := &FP8TextWeights{shards: shards}

	// Load embeddings (full precision)
	embedRaw, embedDtype, embedShape, err := shards.GetRaw("model.decoder.embed_tokens.weight")
	if err != nil {
		return nil, fmt.Errorf("FP8 embed_tokens: %w", err)
	}
	if len(embedShape) != 2 {
		return nil, fmt.Errorf("FP8 embed_tokens shape %v", embedShape)
	}
	out.EmbedShape = [2]int{embedShape[0], embedShape[1]}
	out.EmbedTokens = make([]float32, embedShape[0]*embedShape[1])
	if err := decodeFloatRowTo(out.EmbedTokens, embedRaw, embedDtype); err != nil {
		return nil, fmt.Errorf("FP8 embed_tokens decode: %w", err)
	}

	// Load final norm
	out.FinalNorm, err = loadF32Tensor(shards, "model.decoder.norm.weight")
	if err != nil {
		return nil, fmt.Errorf("FP8 final norm: %w", err)
	}

	// Load self-conditioning
	out.SelfCondPreNorm, err = loadF32Tensor(shards, "model.decoder.self_conditioning.pre_norm.weight")
	if err != nil {
		return nil, fmt.Errorf("FP8 self_cond pre_norm: %w", err)
	}

	// Load layers
	out.Layers = make([]FP8LayerWeights, shape.TextLayers)
	for i := 0; i < shape.TextLayers; i++ {
		lw, err := loadFP8Layer(shards, i)
		if err != nil {
			return nil, fmt.Errorf("FP8 layer %d: %w", i, err)
		}
		out.Layers[i] = lw
	}

	return out, nil
}

func (w *FP8TextWeights) Close() error {
	if w != nil && w.shards != nil {
		return w.shards.Close()
	}
	return nil
}

func loadFP8Layer(shards *safetensors.ShardedFile, layer int) (FP8LayerWeights, error) {
	prefix := fmt.Sprintf("model.decoder.layers.%d", layer)
	lw := FP8LayerWeights{Layer: layer}
	var err error

	// FP8 projections (weight bytes + F32 scale)
	lw.QWeight, lw.QScale, lw.QShape, err = loadFP8Proj(shards, prefix+".self_attn.q_proj")
	if err != nil {
		return lw, err
	}
	lw.KWeight, lw.KScale, lw.KShape, err = loadFP8Proj(shards, prefix+".self_attn.k_proj")
	if err != nil {
		return lw, err
	}
	lw.VWeight, lw.VScale, lw.VShape, err = loadFP8Proj(shards, prefix+".self_attn.v_proj")
	if err != nil {
		// V proj may be absent in full-attention layers (V reuses K)
		lw.VWeight = nil
		lw.VScale = nil
		lw.VShape = lw.KShape
	}
	lw.OWeight, lw.OScale, lw.OShape, err = loadFP8Proj(shards, prefix+".self_attn.o_proj")
	if err != nil {
		return lw, err
	}
	lw.GateWeight, lw.GateScale, lw.GateShape, err = loadFP8Proj(shards, prefix+".mlp.gate_proj")
	if err != nil {
		return lw, err
	}
	lw.UpWeight, lw.UpScale, lw.UpShape, err = loadFP8Proj(shards, prefix+".mlp.up_proj")
	if err != nil {
		return lw, err
	}
	lw.DownWeight, lw.DownScale, lw.DownShape, err = loadFP8Proj(shards, prefix+".mlp.down_proj")
	if err != nil {
		return lw, err
	}

	// Norms (full precision)
	lw.InputNorm, err = loadF32Tensor(shards, prefix+".input_layernorm.weight")
	if err != nil {
		return lw, err
	}
	lw.PostAttnNorm, err = loadF32Tensor(shards, prefix+".post_attention_layernorm.weight")
	if err != nil {
		return lw, err
	}
	lw.PreFFNNorm, err = loadF32Tensor(shards, prefix+".pre_feedforward_layernorm.weight")
	if err != nil {
		return lw, err
	}
	lw.PreFFNNorm2, err = loadF32Tensor(shards, prefix+".pre_feedforward_layernorm_2.weight")
	if err != nil {
		return lw, err
	}
	lw.PostFFNNorm, err = loadF32Tensor(shards, prefix+".post_feedforward_layernorm.weight")
	if err != nil {
		return lw, err
	}
	lw.PostFFNNorm1, err = loadF32Tensor(shards, prefix+".post_feedforward_layernorm_1.weight")
	if err != nil {
		return lw, err
	}
	lw.PostFFNNorm2, err = loadF32Tensor(shards, prefix+".post_feedforward_layernorm_2.weight")
	if err != nil {
		return lw, err
	}
	lw.QNorm, err = loadF32Tensor(shards, prefix+".self_attn.q_norm.weight")
	if err != nil {
		return lw, err
	}
	lw.KNorm, err = loadF32Tensor(shards, prefix+".self_attn.k_norm.weight")
	if err != nil {
		return lw, err
	}

	// Router (full precision)
	lw.RouterScale, err = loadF32Tensor(shards, prefix+".router.scale")
	if err != nil {
		return lw, err
	}
	lw.RouterPerExpertScale, err = loadF32Tensor(shards, prefix+".router.per_expert_scale")
	if err != nil {
		return lw, err
	}
	routerProjRaw, routerDtype, routerShape, err := shards.GetRaw(prefix + ".router.proj.weight")
	if err != nil {
		return lw, err
	}
	if len(routerShape) == 2 {
		lw.RouterProjShape = [2]int{routerShape[0], routerShape[1]}
		lw.RouterProj = make([]float32, routerShape[0]*routerShape[1])
		if err := decodeFloatRowTo(lw.RouterProj, routerProjRaw, routerDtype); err != nil {
			return lw, err
		}
	}

	// Layer scalar
	scalarRaw, scalarDtype, _, err := shards.GetRaw(prefix + ".layer_scalar")
	if err == nil && len(scalarRaw) > 0 {
		var buf [1]float32
		if decodeFloatRowTo(buf[:], scalarRaw, scalarDtype) == nil {
			lw.LayerScalar = buf[0]
		}
	}

	return lw, nil
}

// loadFP8Proj loads FP8 weight bytes (zero-copy from mmap) and F32 scale.
func loadFP8Proj(shards *safetensors.ShardedFile, prefix string) ([]byte, []float32, [2]int, error) {
	weightRaw, _, weightShape, err := shards.GetRaw(prefix + ".weight")
	if err != nil {
		return nil, nil, [2]int{}, fmt.Errorf("FP8 weight %s: %w", prefix, err)
	}
	if len(weightShape) != 2 {
		return nil, nil, [2]int{}, fmt.Errorf("FP8 weight %s shape %v", prefix, weightShape)
	}
	scaleRaw, scaleDtype, scaleShape, err := shards.GetRaw(prefix + ".weight_scale")
	if err != nil {
		return nil, nil, [2]int{}, fmt.Errorf("FP8 scale %s: %w", prefix, err)
	}
	scaleN := 1
	for _, d := range scaleShape {
		scaleN *= d
	}
	scale := make([]float32, scaleN)
	if err := decodeFloatRowTo(scale, scaleRaw, scaleDtype); err != nil {
		return nil, nil, [2]int{}, fmt.Errorf("FP8 scale decode %s: %w", prefix, err)
	}

	// Return mmap slice directly (zero-copy)
	return weightRaw, scale, [2]int{weightShape[0], weightShape[1]}, nil
}

// loadF32Tensor loads a tensor and decodes to F32.
func loadF32Tensor(shards *safetensors.ShardedFile, name string) ([]float32, error) {
	raw, dtype, shape, err := shards.GetRaw(name)
	if err != nil {
		return nil, err
	}
	n := 1
	for _, d := range shape {
		n *= d
	}
	out := make([]float32, n)
	if err := decodeFloatRowTo(out, raw, dtype); err != nil {
		return nil, err
	}
	return out, nil
}

// EagerLoad pre-faults all FP8 safetensor pages into RAM.
func (w *FP8TextWeights) EagerLoad() (int64, error) {
	if w == nil || w.shards == nil {
		return 0, nil
	}
	return w.shards.EagerLoad()
}
