package model

import (
	"encoding/binary"
	"fmt"
	"strings"

	"github.com/rcarmo/go-pherence/backends/mlx"
	"github.com/rcarmo/go-pherence/backends/simd/runtime"
	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/weights"
	"github.com/rcarmo/go-pherence/tensor"
)

// Gemma4MTPDrafter holds the Gemma4 assistant/MTP drafter weights.
//
// The assistant is not a normal Gemma4 decoder: its attention blocks are
// q-only and consume K/V from the main model. It also has pre/post projection
// tensors for activation-conditioned drafting.
type Gemma4MTPDrafter struct {
	Config LlamaConfig

	BackboneHiddenSize int
	NumCentroids       int
	UseOrderedEmbeds   bool

	EmbedTokens              *tensor.Tensor   // [vocab, hidden]
	EmbedTokensMLX           *mlx.QuantWeight // packed [vocab, hidden] for 4-bit assistant weights
	MaskedEmbeddingCentroids *tensor.Tensor   // [numCentroids, hidden]
	MaskedEmbeddingOrdering  []int            // [vocab]

	PreProjection     []float32        // [hidden, 2*backboneHidden]
	PreProjectionMLX  *mlx.QuantWeight // packed [hidden, 2*backboneHidden]
	PostProjection    []float32        // [backboneHidden, hidden]
	PostProjectionMLX *mlx.QuantWeight // packed [backboneHidden, hidden]

	Norm   *tensor.Tensor // [hidden]
	Layers []Gemma4MTPDrafterLayer
}

// Gemma4MTPDrafterLayer is one q-only assistant layer.
type Gemma4MTPDrafterLayer struct {
	InputNorm    *tensor.Tensor
	PostNorm     *tensor.Tensor
	PreFFNNorm   *tensor.Tensor
	PostFFNNorm  *tensor.Tensor
	LayerScalar  float32
	HeadDimLocal int
	// KVSourceLayer is -1 for q-only MTP drafter layers. The drafter
	// forward pass must map each layer to external/main-model K/V state.
	KVSourceLayer int

	QW    []float32        // [numHeads*headDim, hidden]
	QWm   *mlx.QuantWeight // packed [numHeads*headDim, hidden]
	QNorm *tensor.Tensor
	OW    []float32        // [hidden, numHeads*headDim]
	OWm   *mlx.QuantWeight // packed [hidden, numHeads*headDim]

	GateW  []float32        // [intermediate, hidden]
	GateWm *mlx.QuantWeight // packed [intermediate, hidden]
	UpW    []float32        // [intermediate, hidden]
	UpWm   *mlx.QuantWeight // packed [intermediate, hidden]
	DownW  []float32        // [hidden, intermediate]
	DownWm *mlx.QuantWeight // packed [hidden, intermediate]
}

type gemma4AssistantConfig struct {
	Architectures        []string    `json:"architectures"`
	BackboneHiddenSize   int         `json:"backbone_hidden_size"`
	ModelType            string      `json:"model_type"`
	NumCentroids         int         `json:"num_centroids"`
	TextConfig           LlamaConfig `json:"text_config"`
	TieWordEmbeddings    bool        `json:"tie_word_embeddings"`
	UseOrderedEmbeddings bool        `json:"use_ordered_embeddings"`
}

// LoadGemma4MTPDrafter loads a local Gemma4 assistant drafter asset.
func LoadGemma4MTPDrafter(dir string) (*Gemma4MTPDrafter, error) {
	var acfg gemma4AssistantConfig
	cfgData, err := loaderconfig.ReadModelConfig(dir, &acfg)
	if err != nil {
		return nil, err
	}
	if acfg.ModelType != "gemma4_assistant" {
		return nil, fmt.Errorf("expected gemma4_assistant model_type, got %q", acfg.ModelType)
	}

	cfg := acfg.TextConfig
	if cfg.HiddenSize == 0 || cfg.NumLayers == 0 {
		return nil, fmt.Errorf("invalid nested text_config: hidden=%d layers=%d", cfg.HiddenSize, cfg.NumLayers)
	}
	if cfg.VocabSize == 0 {
		return nil, fmt.Errorf("invalid nested text_config: vocab_size=0")
	}
	if cfg.NumHeads == 0 {
		return nil, fmt.Errorf("invalid nested text_config: num_attention_heads=0")
	}
	if cfg.Intermediate == 0 {
		return nil, fmt.Errorf("invalid nested text_config: intermediate_size=0")
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if cfg.HeadDim == 0 {
		if cfg.HiddenSize%cfg.NumHeads != 0 {
			return nil, fmt.Errorf("invalid nested text_config: hidden_size=%d not divisible by num_attention_heads=%d", cfg.HiddenSize, cfg.NumHeads)
		}
		cfg.HeadDim = cfg.HiddenSize / cfg.NumHeads
	}
	if cfg.HiddenAct == "" {
		cfg.HiddenAct = "gelu_pytorch_tanh"
	}
	if quantMeta, err := loaderconfig.ParseQuantizationMetadata(cfgData); err == nil && quantMeta.HasConfig && quantMeta.Bits > 0 {
		cfg.QuantBits = quantMeta.Bits
		cfg.QuantGroup = quantMeta.GroupSize
		cfg.QuantFormat = "mlx"
	}

	f, err := weights.OpenSafetensors(dir)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	loadTensor := func(name string, shape []int) (*tensor.Tensor, error) {
		data, actualShape, err := f.GetFloat32(name)
		if err != nil {
			return nil, fmt.Errorf("load %s: %w", name, err)
		}
		if err := validateShape(name, shape, actualShape, len(data)); err != nil {
			return nil, err
		}
		return tensor.FromFloat32(data, shape), nil
	}
	loadData := func(name string, shape []int) ([]float32, error) {
		data, actualShape, err := f.GetFloat32(name)
		if err != nil {
			return nil, fmt.Errorf("load %s: %w", name, err)
		}
		if err := validateShape(name, shape, actualShape, len(data)); err != nil {
			return nil, err
		}
		return data, nil
	}
	loadDataOrMLX := func(name string, shape []int) ([]float32, *mlx.QuantWeight, error) {
		data, err := loadData(name+".weight", shape)
		if err == nil {
			return data, nil, nil
		}
		if cfg.QuantFormat != "mlx" || cfg.QuantBits <= 0 || cfg.QuantGroup <= 0 || len(shape) != 2 {
			return nil, nil, err
		}
		qw, qerr := mlx.LoadWeight(f, name, shape[0], shape[1], cfg.QuantGroup, cfg.QuantBits)
		if qerr != nil {
			return nil, nil, err
		}
		return nil, qw, nil
	}

	h := cfg.HiddenSize
	d := &Gemma4MTPDrafter{
		Config:             cfg,
		BackboneHiddenSize: acfg.BackboneHiddenSize,
		NumCentroids:       acfg.NumCentroids,
		UseOrderedEmbeds:   acfg.UseOrderedEmbeddings,
		Layers:             make([]Gemma4MTPDrafterLayer, cfg.NumLayers),
	}

	if d.BackboneHiddenSize == 0 {
		return nil, fmt.Errorf("backbone_hidden_size=0")
	}
	if d.NumCentroids == 0 {
		d.NumCentroids = 2048
	}

	if d.EmbedTokens, err = loadTensor("model.embed_tokens.weight", []int{cfg.VocabSize, h}); err != nil {
		if cfg.QuantFormat != "mlx" {
			return nil, err
		}
		emb, qerr := mlx.LoadWeight(f, "model.embed_tokens", cfg.VocabSize, h, cfg.QuantGroup, cfg.QuantBits)
		if qerr != nil {
			return nil, err
		}
		d.EmbedTokensMLX = emb
	}
	if d.UseOrderedEmbeds {
		if d.MaskedEmbeddingCentroids, err = loadTensor("masked_embedding.centroids.weight", []int{d.NumCentroids, h}); err != nil {
			return nil, err
		}
		if d.MaskedEmbeddingOrdering, err = loadIntTensor(f, "masked_embedding.token_ordering", cfg.VocabSize); err != nil {
			return nil, err
		}
	}
	preWidth, ok := checkedProduct(2, d.BackboneHiddenSize)
	if !ok {
		return nil, fmt.Errorf("pre_projection width overflows for backbone_hidden_size=%d", d.BackboneHiddenSize)
	}
	if d.PreProjection, d.PreProjectionMLX, err = loadDataOrMLX("pre_projection", []int{h, preWidth}); err != nil {
		return nil, err
	}
	if d.PostProjection, d.PostProjectionMLX, err = loadDataOrMLX("post_projection", []int{d.BackboneHiddenSize, h}); err != nil {
		return nil, err
	}
	if d.Norm, err = loadTensor("model.norm.weight", []int{h}); err != nil {
		return nil, err
	}

	for l := 0; l < cfg.NumLayers; l++ {
		p := fmt.Sprintf("model.layers.%d", l)
		headDim := cfg.HeadDim
		if l < len(cfg.LayerTypes) && cfg.LayerTypes[l] == "full_attention" && cfg.GlobalHeadDim > 0 {
			headDim = cfg.GlobalHeadDim
		}
		qDim, ok := checkedProduct(cfg.NumHeads, headDim)
		if headDim <= 0 || !ok {
			return nil, fmt.Errorf("drafter layer %d Q dim overflows heads=%d headDim=%d", l, cfg.NumHeads, headDim)
		}

		layer := Gemma4MTPDrafterLayer{
			LayerScalar:   1,
			HeadDimLocal:  headDim,
			KVSourceLayer: -1,
		}
		if layer.InputNorm, err = loadTensor(p+".input_layernorm.weight", []int{h}); err != nil {
			return nil, err
		}
		if layer.PostNorm, err = loadTensor(p+".post_attention_layernorm.weight", []int{h}); err != nil {
			return nil, err
		}
		if layer.PreFFNNorm, err = loadTensor(p+".pre_feedforward_layernorm.weight", []int{h}); err != nil {
			return nil, err
		}
		if layer.PostFFNNorm, err = loadTensor(p+".post_feedforward_layernorm.weight", []int{h}); err != nil {
			return nil, err
		}
		if scalar, err := loadData(p+".layer_scalar", []int{1}); err == nil {
			layer.LayerScalar = scalar[0]
		} else {
			return nil, err
		}

		if layer.QW, layer.QWm, err = loadDataOrMLX(p+".self_attn.q_proj", []int{qDim, h}); err != nil {
			return nil, err
		}
		if layer.QNorm, err = loadTensor(p+".self_attn.q_norm.weight", []int{headDim}); err != nil {
			return nil, err
		}
		if layer.OW, layer.OWm, err = loadDataOrMLX(p+".self_attn.o_proj", []int{h, qDim}); err != nil {
			return nil, err
		}

		if layer.GateW, layer.GateWm, err = loadDataOrMLX(p+".mlp.gate_proj", []int{cfg.Intermediate, h}); err != nil {
			return nil, err
		}
		if layer.UpW, layer.UpWm, err = loadDataOrMLX(p+".mlp.up_proj", []int{cfg.Intermediate, h}); err != nil {
			return nil, err
		}
		if layer.DownW, layer.DownWm, err = loadDataOrMLX(p+".mlp.down_proj", []int{h, cfg.Intermediate}); err != nil {
			return nil, err
		}

		d.Layers[l] = layer
	}

	return d, nil
}

// AssistantTokenEmbeddingInto copies the assistant/drafter embedding row for tokenID.
func (d *Gemma4MTPDrafter) AssistantTokenEmbeddingInto(dst []float32, tokenID int) error {
	if d == nil || (d.EmbedTokens == nil && d.EmbedTokensMLX == nil) {
		return fmt.Errorf("drafter embeddings are not loaded")
	}
	h := d.Config.HiddenSize
	if len(dst) != h {
		return fmt.Errorf("assistant token embedding dst len=%d, want %d", len(dst), h)
	}
	if tokenID < 0 || tokenID >= d.Config.VocabSize {
		return fmt.Errorf("token id %d out of range [0,%d)", tokenID, d.Config.VocabSize)
	}
	if d.EmbedTokensMLX != nil {
		if !mlx.DequantRowTo(dst, d.EmbedTokensMLX, tokenID) {
			return fmt.Errorf("assistant MLX embedding row %d dequant failed", tokenID)
		}
		return nil
	}
	emb := d.EmbedTokens.Data()
	need := (tokenID + 1) * h
	if len(emb) < need {
		return fmt.Errorf("assistant embedding data len=%d, want at least %d", len(emb), need)
	}
	copy(dst, emb[tokenID*h:need])
	return nil
}

// MaskedEmbeddingOrder returns the masked-embedding ordering entry for tokenID.
func (d *Gemma4MTPDrafter) MaskedEmbeddingOrder(tokenID int) (int, error) {
	if d == nil || d.MaskedEmbeddingOrdering == nil {
		return 0, fmt.Errorf("masked embedding ordering is not loaded")
	}
	if tokenID < 0 || tokenID >= len(d.MaskedEmbeddingOrdering) {
		return 0, fmt.Errorf("token id %d out of range [0,%d)", tokenID, len(d.MaskedEmbeddingOrdering))
	}
	return d.MaskedEmbeddingOrdering[tokenID], nil
}

// PreProjectInto computes dst = pre_projection · (backboneTokenEmbedding || activation).
// Both inputs are main/backbone-width vectors, not assistant hidden-size vectors.
func (d *Gemma4MTPDrafter) PreProjectInto(dst, backboneTokenEmbedding, activation []float32) error {
	if d == nil {
		return fmt.Errorf("nil drafter")
	}
	bh := d.BackboneHiddenSize
	h := d.Config.HiddenSize
	if h <= 0 || bh <= 0 {
		return fmt.Errorf("invalid projection dims hidden=%d backbone=%d", h, bh)
	}
	preWidth, ok := checkedProduct(2, bh)
	if !ok {
		return fmt.Errorf("pre-project width overflows for backbone=%d", bh)
	}
	want, ok := checkedProduct(h, preWidth)
	if !ok {
		return fmt.Errorf("pre-project size overflows hidden=%d backbone=%d", h, bh)
	}
	if len(dst) != h {
		return fmt.Errorf("pre-project dst len=%d, want %d", len(dst), h)
	}
	if len(backboneTokenEmbedding) != bh {
		return fmt.Errorf("pre-project token embedding len=%d, want %d", len(backboneTokenEmbedding), bh)
	}
	if len(activation) != bh {
		return fmt.Errorf("pre-project activation len=%d, want %d", len(activation), bh)
	}
	if d.PreProjectionMLX != nil {
		in := make([]float32, preWidth)
		copy(in, backboneTokenEmbedding)
		copy(in[bh:], activation)
		if !mlx.GemvTo(dst, in, d.PreProjectionMLX) {
			return fmt.Errorf("pre_projection MLX GEMV failed")
		}
		return nil
	}
	if len(d.PreProjection) < want {
		return fmt.Errorf("pre_projection len=%d, want at least %d", len(d.PreProjection), want)
	}
	out := make([]float32, h)
	for row := 0; row < h; row++ {
		start := row * preWidth
		w := d.PreProjection[start : start+preWidth]
		out[row] = simdDot(backboneTokenEmbedding, w[:bh]) + simdDot(activation, w[bh:])
	}
	copy(dst, out)
	return nil
}

// PostProjectInto computes dst = post_projection · assistantHidden.
func (d *Gemma4MTPDrafter) PostProjectInto(dst, assistantHidden []float32) error {
	if d == nil {
		return fmt.Errorf("nil drafter")
	}
	bh := d.BackboneHiddenSize
	h := d.Config.HiddenSize
	if h <= 0 || bh <= 0 {
		return fmt.Errorf("invalid projection dims hidden=%d backbone=%d", h, bh)
	}
	want, ok := checkedProduct(bh, h)
	if !ok {
		return fmt.Errorf("post-project size overflows hidden=%d backbone=%d", h, bh)
	}
	if len(dst) != bh {
		return fmt.Errorf("post-project dst len=%d, want %d", len(dst), bh)
	}
	if len(assistantHidden) != h {
		return fmt.Errorf("post-project hidden len=%d, want %d", len(assistantHidden), h)
	}
	if d.PostProjectionMLX != nil {
		if !mlx.GemvTo(dst, assistantHidden, d.PostProjectionMLX) {
			return fmt.Errorf("post_projection MLX GEMV failed")
		}
		return nil
	}
	if len(d.PostProjection) < want {
		return fmt.Errorf("post_projection len=%d, want at least %d", len(d.PostProjection), want)
	}
	out := make([]float32, bh)
	gemvNT(out, assistantHidden, d.PostProjection, h, bh)
	copy(dst, out)
	return nil
}

func simdDot(a, b []float32) float32 {
	return simd.Sdot(a, b)
}

func validateShape(name string, expected, actual []int, n int) error {
	expectedN := shapeProduct(expected)
	if expectedN < 0 {
		return fmt.Errorf("load %s: invalid expected shape %v", name, expected)
	}
	if len(actual) == 0 {
		if expectedN != n {
			return fmt.Errorf("load %s: shape unavailable, expected %v (%d elems), got %d elems", name, expected, expectedN, n)
		}
		return nil
	}
	actualN := shapeProduct(actual)
	if actualN < 0 {
		return fmt.Errorf("load %s: invalid actual shape %v", name, actual)
	}
	if !sameShape(actual, expected) {
		return fmt.Errorf("load %s: shape mismatch: expected %v, actual %v", name, expected, actual)
	}
	if actualN != n {
		return fmt.Errorf("load %s: shape %v has %d elems, data has %d", name, actual, actualN, n)
	}
	return nil
}

func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func shapeProduct(shape []int) int {
	prod := 1
	maxInt := int(^uint(0) >> 1)
	for _, dim := range shape {
		if dim < 0 || (dim > 0 && prod > maxInt/dim) {
			return -1
		}
		prod *= dim
	}
	return prod
}

func loadIntTensor(f weights.Source, name string, expectedLen int) ([]int, error) {
	raw, dtype, shape, err := f.GetRaw(name)
	if err != nil {
		return nil, fmt.Errorf("load %s: %w", name, err)
	}
	if expectedLen < 0 {
		return nil, fmt.Errorf("load %s: invalid expected length %d", name, expectedLen)
	}
	if err := validateShape(name, []int{expectedLen}, shape, expectedLen); err != nil {
		return nil, err
	}
	out := make([]int, expectedLen)
	maxInt := int64(int(^uint(0) >> 1))
	minInt := -maxInt - 1
	switch strings.ToUpper(dtype) {
	case "I64", "INT64":
		wantBytes, ok := checkedProduct(expectedLen, 8)
		if !ok {
			return nil, fmt.Errorf("load %s: expected I64 byte size overflows for len %d", name, expectedLen)
		}
		if len(raw) != wantBytes {
			return nil, fmt.Errorf("load %s: raw size %d does not match I64 len %d", name, len(raw), expectedLen)
		}
		for i := range out {
			v := int64(binary.LittleEndian.Uint64(raw[i*8:]))
			if v < minInt || v > maxInt {
				return nil, fmt.Errorf("load %s: value %d overflows int", name, v)
			}
			out[i] = int(v)
		}
	case "I32", "INT32":
		wantBytes, ok := checkedProduct(expectedLen, 4)
		if !ok {
			return nil, fmt.Errorf("load %s: expected I32 byte size overflows for len %d", name, expectedLen)
		}
		if len(raw) != wantBytes {
			return nil, fmt.Errorf("load %s: raw size %d does not match I32 len %d", name, len(raw), expectedLen)
		}
		for i := range out {
			out[i] = int(int32(binary.LittleEndian.Uint32(raw[i*4:])))
		}
	default:
		return nil, fmt.Errorf("load %s: unsupported integer dtype %s", name, dtype)
	}
	return out, nil
}
