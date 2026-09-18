package gliner2

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/rcarmo/go-pherence/internal/checked"
)

var requiredDebertaConfigFields = []string{
	"hidden_act",
	"hidden_size",
	"intermediate_size",
	"layer_norm_eps",
	"max_position_embeddings",
	"max_relative_positions",
	"norm_rel_ebd",
	"num_attention_heads",
	"num_hidden_layers",
	"pos_att_type",
	"position_biased_input",
	"position_buckets",
	"relative_attention",
	"share_att_key",
	"type_vocab_size",
	"vocab_size",
}

// DebertaConfig is the supported published GLiNER2.5 encoder subset: plain
// word embeddings, relative attention with shared query/key projections, and
// exact GELU feed-forward blocks.
type DebertaConfig struct {
	HiddenAct             string                     `json:"hidden_act"`
	HiddenSize            int                        `json:"hidden_size"`
	IntermediateSize      int                        `json:"intermediate_size"`
	LayerNormEps          float32                    `json:"layer_norm_eps"`
	MaxPositionEmbeddings int                        `json:"max_position_embeddings"`
	MaxRelativePositions  int                        `json:"max_relative_positions"`
	NormRelEbd            string                     `json:"norm_rel_ebd"`
	NumAttentionHeads     int                        `json:"num_attention_heads"`
	NumHiddenLayers       int                        `json:"num_hidden_layers"`
	PosAttType            []string                   `json:"pos_att_type"`
	PositionBiasedInput   bool                       `json:"position_biased_input"`
	PositionBuckets       int                        `json:"position_buckets"`
	RelativeAttention     bool                       `json:"relative_attention"`
	ShareAttKey           bool                       `json:"share_att_key"`
	TypeVocabSize         int                        `json:"type_vocab_size"`
	VocabSize             int                        `json:"vocab_size"`
	Raw                   map[string]json.RawMessage `json:"-"`
}

func (c *DebertaConfig) UnmarshalJSON(data []byte) error {
	type rawDebertaConfig DebertaConfig
	var rawCfg rawDebertaConfig
	raw, err := decodeObject(json.RawMessage(data), "deberta_config", &rawCfg)
	if err != nil {
		return err
	}
	if err := requireFields(raw, "deberta_config", requiredDebertaConfigFields); err != nil {
		return err
	}
	cfg := DebertaConfig(rawCfg).normalized()
	cfg.Raw = raw
	if err := cfg.Validate(); err != nil {
		return err
	}
	*c = cfg
	return nil
}

func (c DebertaConfig) normalized() DebertaConfig {
	out := c
	out.HiddenAct = normalizeLower(out.HiddenAct)
	out.NormRelEbd = normalizeLower(out.NormRelEbd)
	if len(out.PosAttType) != 0 {
		norm := make([]string, len(out.PosAttType))
		for i, item := range out.PosAttType {
			norm[i] = normalizeLower(item)
		}
		out.PosAttType = norm
	}
	return out
}

func (c DebertaConfig) maxRelativePosition() int {
	if c.MaxRelativePositions > 0 {
		return c.MaxRelativePositions
	}
	return c.MaxPositionEmbeddings
}

func (c DebertaConfig) relativeEmbeddingSpan() int {
	if c.PositionBuckets > 0 {
		return c.PositionBuckets
	}
	return c.maxRelativePosition()
}

func (c DebertaConfig) Validate() error {
	c = c.normalized()
	if c.HiddenSize <= 0 || c.NumAttentionHeads <= 0 || c.NumHiddenLayers <= 0 || c.IntermediateSize <= 0 || c.VocabSize <= 0 {
		return fmt.Errorf("deberta_config invalid dims hidden=%d heads=%d layers=%d intermediate=%d vocab=%d", c.HiddenSize, c.NumAttentionHeads, c.NumHiddenLayers, c.IntermediateSize, c.VocabSize)
	}
	if c.HiddenSize%c.NumAttentionHeads != 0 {
		return fmt.Errorf("deberta_config hidden_size=%d not divisible by num_attention_heads=%d", c.HiddenSize, c.NumAttentionHeads)
	}
	if c.LayerNormEps <= 0 || math.IsNaN(float64(c.LayerNormEps)) || math.IsInf(float64(c.LayerNormEps), 0) {
		return fmt.Errorf("deberta_config layer_norm_eps=%g", c.LayerNormEps)
	}
	if c.MaxPositionEmbeddings <= 0 {
		return fmt.Errorf("deberta_config max_position_embeddings=%d", c.MaxPositionEmbeddings)
	}
	if c.PositionBuckets <= 0 || c.PositionBuckets%2 != 0 {
		return fmt.Errorf("deberta_config position_buckets=%d must be positive and even", c.PositionBuckets)
	}
	if !c.RelativeAttention {
		return fmt.Errorf("deberta_config relative_attention must be true")
	}
	if !c.ShareAttKey {
		return fmt.Errorf("deberta_config share_att_key must be true")
	}
	if c.PositionBiasedInput {
		return fmt.Errorf("deberta_config position_biased_input must be false")
	}
	if c.TypeVocabSize != 0 {
		return fmt.Errorf("deberta_config type_vocab_size=%d must be 0", c.TypeVocabSize)
	}
	if c.HiddenAct != "gelu" {
		return fmt.Errorf("deberta_config hidden_act=%q must be \"gelu\"", c.HiddenAct)
	}
	if c.NormRelEbd != "layer_norm" {
		return fmt.Errorf("deberta_config norm_rel_ebd=%q must be \"layer_norm\"", c.NormRelEbd)
	}
	if len(c.PosAttType) != 2 {
		return fmt.Errorf("deberta_config pos_att_type=%v want [c2p p2c]", c.PosAttType)
	}
	seenC2P, seenP2C := false, false
	for _, item := range c.PosAttType {
		switch item {
		case "c2p":
			if seenC2P {
				return fmt.Errorf("deberta_config pos_att_type=%v duplicates c2p", c.PosAttType)
			}
			seenC2P = true
		case "p2c":
			if seenP2C {
				return fmt.Errorf("deberta_config pos_att_type=%v duplicates p2c", c.PosAttType)
			}
			seenP2C = true
		default:
			return fmt.Errorf("deberta_config pos_att_type=%v must contain only c2p and p2c", c.PosAttType)
		}
	}
	if !seenC2P || !seenP2C {
		return fmt.Errorf("deberta_config pos_att_type=%v want both c2p and p2c", c.PosAttType)
	}
	if maxRelative := c.maxRelativePosition(); maxRelative <= c.PositionBuckets/2+1 {
		return fmt.Errorf("deberta_config max_relative_positions/max_position_embeddings=%d incompatible with position_buckets=%d", maxRelative, c.PositionBuckets)
	}
	return nil
}

// DebertaLayer is one published DeBERTa-v2 encoder block.
type DebertaLayer struct {
	QueryProjection    Linear    `json:"query_projection"`
	KeyProjection      Linear    `json:"key_projection"`
	ValueProjection    Linear    `json:"value_projection"`
	AttentionOutput    Linear    `json:"attention_output"`
	AttentionLayerNorm LayerNorm `json:"attention_layer_norm"`
	IntermediateDense  Linear    `json:"intermediate_dense"`
	OutputDense        Linear    `json:"output_dense"`
	OutputLayerNorm    LayerNorm `json:"output_layer_norm"`
}

func (l DebertaLayer) Validate(hiddenSize, intermediateSize int, eps float32) error {
	for name, projection := range map[string]Linear{
		"query_projection":   l.QueryProjection,
		"key_projection":     l.KeyProjection,
		"value_projection":   l.ValueProjection,
		"attention_output":   l.AttentionOutput,
		"intermediate_dense": l.IntermediateDense,
		"output_dense":       l.OutputDense,
	} {
		if err := projection.Validate(); err != nil {
			return fmt.Errorf("deberta layer %s: %w", name, err)
		}
	}
	for name, norm := range map[string]LayerNorm{
		"attention_layer_norm": l.AttentionLayerNorm,
		"output_layer_norm":    l.OutputLayerNorm,
	} {
		if err := validateDebertaNorm(name, norm, hiddenSize, eps); err != nil {
			return err
		}
	}
	for name, projection := range map[string]Linear{
		"query_projection": l.QueryProjection,
		"key_projection":   l.KeyProjection,
		"value_projection": l.ValueProjection,
		"attention_output": l.AttentionOutput,
	} {
		if projection.InDim != hiddenSize || projection.OutDim != hiddenSize {
			return fmt.Errorf("deberta layer %s dims out=%d in=%d want out=%d in=%d", name, projection.OutDim, projection.InDim, hiddenSize, hiddenSize)
		}
	}
	if l.IntermediateDense.InDim != hiddenSize || l.IntermediateDense.OutDim != intermediateSize {
		return fmt.Errorf("deberta layer intermediate_dense dims out=%d in=%d want out=%d in=%d", l.IntermediateDense.OutDim, l.IntermediateDense.InDim, intermediateSize, hiddenSize)
	}
	if l.OutputDense.InDim != intermediateSize || l.OutputDense.OutDim != hiddenSize {
		return fmt.Errorf("deberta layer output_dense dims out=%d in=%d want out=%d in=%d", l.OutputDense.OutDim, l.OutputDense.InDim, hiddenSize, intermediateSize)
	}
	return nil
}

// Deberta is the published GLiNER2.5 encoder subset: word embeddings followed
// by DeBERTa-v2 relative self-attention layers.
type Deberta struct {
	Config              DebertaConfig  `json:"config"`
	WordEmbeddings      []float32      `json:"word_embeddings"`
	EmbeddingsLayerNorm LayerNorm      `json:"embeddings_layer_norm"`
	RelativeEmbeddings  []float32      `json:"relative_embeddings"`
	RelativeLayerNorm   LayerNorm      `json:"relative_layer_norm"`
	Layers              []DebertaLayer `json:"layers"`
}

func validateDebertaNorm(name string, norm LayerNorm, dim int, eps float32) error {
	if err := norm.Validate(); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	if norm.Dim() != dim {
		return fmt.Errorf("%s dim=%d want=%d", name, norm.Dim(), dim)
	}
	if norm.epsilon() != eps {
		return fmt.Errorf("%s epsilon=%g want=%g", name, norm.epsilon(), eps)
	}
	return nil
}

func (m Deberta) Validate() error {
	cfg := m.Config.normalized()
	if err := cfg.Validate(); err != nil {
		return err
	}
	wordLen, ok := checked.MulInt(cfg.VocabSize, cfg.HiddenSize)
	if !ok {
		return fmt.Errorf("deberta word embeddings shape overflow vocab=%d hidden=%d", cfg.VocabSize, cfg.HiddenSize)
	}
	if len(m.WordEmbeddings) != wordLen {
		return fmt.Errorf("deberta word_embeddings len=%d want=%d", len(m.WordEmbeddings), wordLen)
	}
	if err := validateDebertaNorm("embeddings_layer_norm", m.EmbeddingsLayerNorm, cfg.HiddenSize, cfg.LayerNormEps); err != nil {
		return err
	}
	relRows, ok := checked.MulInt(2, cfg.relativeEmbeddingSpan())
	if !ok {
		return fmt.Errorf("deberta relative embedding rows overflow span=%d", cfg.relativeEmbeddingSpan())
	}
	relLen, ok := checked.MulInt(relRows, cfg.HiddenSize)
	if !ok {
		return fmt.Errorf("deberta relative embeddings shape overflow rows=%d hidden=%d", relRows, cfg.HiddenSize)
	}
	if len(m.RelativeEmbeddings) != relLen {
		return fmt.Errorf("deberta relative_embeddings len=%d want=%d", len(m.RelativeEmbeddings), relLen)
	}
	if err := validateDebertaNorm("relative_layer_norm", m.RelativeLayerNorm, cfg.HiddenSize, cfg.LayerNormEps); err != nil {
		return err
	}
	if len(m.Layers) != cfg.NumHiddenLayers {
		return fmt.Errorf("deberta layers=%d want=%d", len(m.Layers), cfg.NumHiddenLayers)
	}
	for i, layer := range m.Layers {
		if err := layer.Validate(cfg.HiddenSize, cfg.IntermediateSize, cfg.LayerNormEps); err != nil {
			return fmt.Errorf("deberta layer[%d]: %w", i, err)
		}
	}
	return nil
}

// LoadDeberta binds the published GLiNER2.5 encoder tensors from the exact
// encoder.* safetensors prefixes.
func LoadDeberta(source TensorSource, cfg DebertaConfig) (Deberta, error) {
	var m Deberta
	if source == nil {
		return m, fmt.Errorf("deberta tensor source required")
	}
	cfg = cfg.normalized()
	if err := cfg.Validate(); err != nil {
		return m, err
	}
	norm := func(r *weightReader, name string, dim int) LayerNorm {
		return LayerNorm{Weight: r.tensor(name+".weight", dim), Bias: r.tensor(name+".bias", dim), Epsilon: cfg.LayerNormEps}
	}
	r := weightReader{source: source}
	relRows := 2 * cfg.relativeEmbeddingSpan()
	m = Deberta{
		Config:              cfg,
		WordEmbeddings:      r.tensor("encoder.embeddings.word_embeddings.weight", cfg.VocabSize, cfg.HiddenSize),
		EmbeddingsLayerNorm: norm(&r, "encoder.embeddings.LayerNorm", cfg.HiddenSize),
		RelativeEmbeddings:  r.tensor("encoder.encoder.rel_embeddings.weight", relRows, cfg.HiddenSize),
		RelativeLayerNorm:   norm(&r, "encoder.encoder.LayerNorm", cfg.HiddenSize),
	}
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		prefix := fmt.Sprintf("encoder.encoder.layer.%d", i)
		m.Layers = append(m.Layers, DebertaLayer{
			QueryProjection:    r.linear(prefix+".attention.self.query_proj", cfg.HiddenSize, cfg.HiddenSize),
			KeyProjection:      r.linear(prefix+".attention.self.key_proj", cfg.HiddenSize, cfg.HiddenSize),
			ValueProjection:    r.linear(prefix+".attention.self.value_proj", cfg.HiddenSize, cfg.HiddenSize),
			AttentionOutput:    r.linear(prefix+".attention.output.dense", cfg.HiddenSize, cfg.HiddenSize),
			AttentionLayerNorm: norm(&r, prefix+".attention.output.LayerNorm", cfg.HiddenSize),
			IntermediateDense:  r.linear(prefix+".intermediate.dense", cfg.HiddenSize, cfg.IntermediateSize),
			OutputDense:        r.linear(prefix+".output.dense", cfg.IntermediateSize, cfg.HiddenSize),
			OutputLayerNorm:    norm(&r, prefix+".output.LayerNorm", cfg.HiddenSize),
		})
	}
	if r.err != nil {
		return Deberta{}, r.err
	}
	return m, m.Validate()
}

// Encode embeds one token-id sequence and runs the supported DeBERTa encoder.
// Token masking matches the published model: embedding outputs are masked once
// before the first layer and attention visibility uses the same validity mask.
func (m Deberta) Encode(tokenIDs []int, mask []bool) ([][]float32, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	if len(tokenIDs) != len(mask) {
		return nil, fmt.Errorf("deberta mask len=%d want=%d", len(mask), len(tokenIDs))
	}
	rows := len(tokenIDs)
	if rows == 0 {
		return [][]float32{}, nil
	}
	cfg := m.Config
	hidden := cfg.HiddenSize
	stateLen, ok := checked.MulInt(rows, hidden)
	if !ok {
		return nil, fmt.Errorf("deberta state shape overflow rows=%d hidden=%d", rows, hidden)
	}
	statesIn := make([]float32, stateLen)
	for i, tokenID := range tokenIDs {
		if tokenID < 0 || tokenID >= cfg.VocabSize {
			return nil, fmt.Errorf("deberta token_ids[%d]=%d outside [0,%d)", i, tokenID, cfg.VocabSize)
		}
		copy(statesIn[i*hidden:(i+1)*hidden], m.WordEmbeddings[tokenID*hidden:(tokenID+1)*hidden])
	}
	states := make([]float32, stateLen)
	if err := m.EmbeddingsLayerNorm.ApplyBatch(statesIn, states, rows); err != nil {
		return nil, err
	}
	for i, valid := range mask {
		if !valid {
			clear(states[i*hidden : (i+1)*hidden])
		}
	}
	if len(m.Layers) == 0 {
		return rowsFromFlat(states, rows, hidden), nil
	}
	relRows := 2 * cfg.relativeEmbeddingSpan()
	relNormed := make([]float32, len(m.RelativeEmbeddings))
	if err := m.RelativeLayerNorm.ApplyBatch(m.RelativeEmbeddings, relNormed, relRows); err != nil {
		return nil, err
	}
	relProjLen := len(relNormed)
	ffnLen, ok := checked.MulInt(rows, cfg.IntermediateSize)
	if !ok {
		return nil, fmt.Errorf("deberta ffn shape overflow rows=%d intermediate=%d", rows, cfg.IntermediateSize)
	}
	for i, layer := range m.Layers {
		q := make([]float32, stateLen)
		k := make([]float32, stateLen)
		v := make([]float32, stateLen)
		if err := layer.QueryProjection.ApplyBatch(states, q, rows); err != nil {
			return nil, fmt.Errorf("deberta layer %d query: %w", i, err)
		}
		if err := layer.KeyProjection.ApplyBatch(states, k, rows); err != nil {
			return nil, fmt.Errorf("deberta layer %d key: %w", i, err)
		}
		if err := layer.ValueProjection.ApplyBatch(states, v, rows); err != nil {
			return nil, fmt.Errorf("deberta layer %d value: %w", i, err)
		}
		relQ := make([]float32, relProjLen)
		relK := make([]float32, relProjLen)
		if err := layer.QueryProjection.ApplyBatch(relNormed, relQ, relRows); err != nil {
			return nil, fmt.Errorf("deberta layer %d relative query: %w", i, err)
		}
		if err := layer.KeyProjection.ApplyBatch(relNormed, relK, relRows); err != nil {
			return nil, fmt.Errorf("deberta layer %d relative key: %w", i, err)
		}
		context, err := DisentangledAttention(
			debertaRowsView(q, rows, hidden),
			debertaRowsView(k, rows, hidden),
			debertaRowsView(v, rows, hidden),
			debertaRowsView(relQ, relRows, hidden),
			debertaRowsView(relK, relRows, hidden),
			mask,
			cfg.NumAttentionHeads,
			cfg.PositionBuckets,
			cfg.maxRelativePosition(),
		)
		if err != nil {
			return nil, fmt.Errorf("deberta layer %d attention: %w", i, err)
		}
		contextFlat, err := flattenRows(context, hidden, fmt.Sprintf("deberta layer %d context", i))
		if err != nil {
			return nil, err
		}
		attnUpdate := make([]float32, stateLen)
		if err := layer.AttentionOutput.ApplyBatch(contextFlat, attnUpdate, rows); err != nil {
			return nil, fmt.Errorf("deberta layer %d attention output: %w", i, err)
		}
		attnResidual := make([]float32, stateLen)
		for j := range attnResidual {
			attnResidual[j] = states[j] + attnUpdate[j]
		}
		attnStates := make([]float32, stateLen)
		if err := layer.AttentionLayerNorm.ApplyBatch(attnResidual, attnStates, rows); err != nil {
			return nil, fmt.Errorf("deberta layer %d attention layer_norm: %w", i, err)
		}
		ffnHidden := make([]float32, ffnLen)
		if err := layer.IntermediateDense.ApplyBatch(attnStates, ffnHidden, rows); err != nil {
			return nil, fmt.Errorf("deberta layer %d intermediate: %w", i, err)
		}
		for j := range ffnHidden {
			ffnHidden[j] = gelu32(ffnHidden[j])
		}
		ffnUpdate := make([]float32, stateLen)
		if err := layer.OutputDense.ApplyBatch(ffnHidden, ffnUpdate, rows); err != nil {
			return nil, fmt.Errorf("deberta layer %d output dense: %w", i, err)
		}
		ffnResidual := make([]float32, stateLen)
		for j := range ffnResidual {
			ffnResidual[j] = attnStates[j] + ffnUpdate[j]
		}
		next := make([]float32, stateLen)
		if err := layer.OutputLayerNorm.ApplyBatch(ffnResidual, next, rows); err != nil {
			return nil, fmt.Errorf("deberta layer %d output layer_norm: %w", i, err)
		}
		states = next
	}
	return rowsFromFlat(states, rows, hidden), nil
}

func debertaRowsView(flat []float32, rows, cols int) [][]float32 {
	out := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		out[i] = flat[i*cols : (i+1)*cols]
	}
	return out
}

func (c DebertaConfig) String() string {
	return fmt.Sprintf("hidden=%d heads=%d layers=%d intermediate=%d vocab=%d act=%s rel=%t share=%t pos_att=%s", c.HiddenSize, c.NumAttentionHeads, c.NumHiddenLayers, c.IntermediateSize, c.VocabSize, c.HiddenAct, c.RelativeAttention, c.ShareAttKey, strings.Join(c.PosAttType, "+"))
}
