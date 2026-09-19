package jevlike

import (
	"fmt"
	"strings"

	backbone "github.com/rcarmo/go-pherence/model"
)

// TokenEncoder returns final hidden states without an LM-head projection.
// Implementations must encode special tokens and truncate according to their
// own tokenizer contract. They must not mutate model weights.
type TokenEncoder interface {
	Encode(text string, maxTokens int) ([][]float32, error)
}

// PooledOptionEncoder stores deduplicated option vectors after F32 pooling,
// avoiding re-pooling rounded token states on every cached epoch.
type PooledOptionEncoder interface {
	EncodeOption(text string, maxTokens int) ([]float32, error)
}

// DecoderEncoder adapts a loaded go-pherence dense causal decoder. Tokenize is
// explicit: callers own BOS/EOS policy and must match the checkpoint tokenizer.
// A decoder instance must not be shared with concurrent generation.
type DecoderEncoder struct {
	Model    *backbone.LlamaModel
	Tokenize func(string) ([]int, error)
}

func (e DecoderEncoder) Encode(text string, maxTokens int) ([][]float32, error) {
	if e.Model == nil || e.Tokenize == nil || maxTokens <= 0 {
		return nil, fmt.Errorf("invalid frozen decoder encoder")
	}
	ids, err := e.Tokenize(text)
	if err != nil {
		return nil, err
	}
	if len(ids) > maxTokens {
		ids = ids[:maxTokens]
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("encoder produced no tokens")
	}
	return e.Model.EncodeTokenHiddenStates(ids)
}

// FrozenScorer owns only a trainable head. The encoder is supplied separately,
// not serialized or updated as part of a scorer checkpoint.
type FrozenScorer struct {
	Config    Config
	Reference string
	Head      AttentionHead
	Encoder   TokenEncoder
}

type frozenEncodedBatch struct {
	Context     [][][]float32
	ContextMask [][]bool
	Options     [][][]float32
	OptionMask  [][]bool
	Labels      []int
}

func (c Checkpoint) Frozen(encoder TokenEncoder) (*FrozenScorer, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	if c.Encoder != "frozen" || encoder == nil {
		return nil, fmt.Errorf("frozen checkpoint and encoder required")
	}
	if strings.HasPrefix(c.EncoderReference, "feature:") {
		identified, ok := encoder.(interface{ FeatureReference() string })
		if !ok || identified.FeatureReference() != c.EncoderReference {
			return nil, fmt.Errorf("exact feature encoder identity required")
		}
	}
	h, err := NewAttentionHead(c.Config.Width, c.Config.Rank)
	if err != nil {
		return nil, err
	}
	params := make(map[string][]float32)
	for _, p := range c.Parameters {
		params[p.Name] = p.Values
	}
	if err = h.LoadNamedParameters(defaultHeadPrefix, params); err != nil {
		return nil, err
	}
	return &FrozenScorer{Config: c.Config, Reference: c.EncoderReference, Head: *h, Encoder: encoder}, nil
}

func FrozenCheckpoint(m *FrozenScorer) (Checkpoint, error) {
	if err := validateFrozenScorer(m, false); err != nil {
		return Checkpoint{}, err
	}
	if m.Reference == "" {
		return Checkpoint{}, fmt.Errorf("frozen checkpoint requires encoder_reference")
	}
	return Checkpoint{
		Version:          1,
		Encoder:          "frozen",
		EncoderReference: m.Reference,
		Config:           m.Config,
		Parameters:       m.Head.NamedParameters(defaultHeadPrefix),
	}, nil
}

func (m *FrozenScorer) Forward(examples []ChoiceExample, shuffle bool) ([][]float32, error) {
	batch, err := m.encodeBatch(examples)
	if err != nil {
		return nil, err
	}
	return m.forwardEncodedBatch(batch, shuffle)
}

func (m *FrozenScorer) encodeBatch(examples []ChoiceExample) (frozenEncodedBatch, error) {
	if err := validateFrozenScorer(m, true); err != nil {
		return frozenEncodedBatch{}, err
	}
	if len(examples) == 0 {
		return frozenEncodedBatch{}, fmt.Errorf("empty batch")
	}

	validated := make([]ChoiceExample, len(examples))
	contexts := make([][][]float32, len(examples))
	options := make([][][]float32, len(examples))
	maxCtx, maxOpts := 0, 0
	for i, ex := range examples {
		item, err := ValidateChoiceExample(ex)
		if err != nil {
			return frozenEncodedBatch{}, err
		}
		validated[i] = item

		ctx, err := m.Encoder.Encode(item.Context, m.Config.ContextTokens)
		if err != nil {
			return frozenEncodedBatch{}, err
		}
		if err = validateEncoded(ctx, m.Config.Width); err != nil {
			return frozenEncodedBatch{}, err
		}
		contexts[i] = ctx
		maxCtx = max(maxCtx, len(ctx))
		maxOpts = max(maxOpts, len(item.Options))

		options[i] = make([][]float32, len(item.Options))
		for j, text := range item.Options {
			if encoder, ok := m.Encoder.(PooledOptionEncoder); ok {
				pooled, err := encoder.EncodeOption(text, m.Config.OptionTokens)
				if err != nil {
					return frozenEncodedBatch{}, err
				}
				if err = validateFeatureRows([][]float32{pooled}, m.Config.Width); err != nil {
					return frozenEncodedBatch{}, err
				}
				options[i][j] = pooled
				continue
			}
			tokens, err := m.Encoder.Encode(text, m.Config.OptionTokens)
			if err != nil {
				return frozenEncodedBatch{}, err
			}
			if err = validateEncoded(tokens, m.Config.Width); err != nil {
				return frozenEncodedBatch{}, err
			}
			pooled := make([]float32, m.Config.Width)
			for _, row := range tokens {
				for d, v := range row {
					pooled[d] += v / float32(len(tokens))
				}
			}
			options[i][j] = pooled
		}
	}

	batch := frozenEncodedBatch{
		Context:     contexts,
		ContextMask: make([][]bool, len(validated)),
		Options:     options,
		OptionMask:  make([][]bool, len(validated)),
		Labels:      make([]int, len(validated)),
	}
	for i, item := range validated {
		batch.ContextMask[i] = make([]bool, maxCtx)
		for j := range contexts[i] {
			batch.ContextMask[i][j] = true
		}
		for len(contexts[i]) < maxCtx {
			contexts[i] = append(contexts[i], make([]float32, m.Config.Width))
		}
		batch.Context[i] = contexts[i]

		batch.OptionMask[i] = make([]bool, maxOpts)
		for j := range options[i] {
			batch.OptionMask[i][j] = true
		}
		for len(options[i]) < maxOpts {
			options[i] = append(options[i], make([]float32, m.Config.Width))
		}
		batch.Options[i] = options[i]
		batch.Labels[i] = item.Label
	}
	return batch, nil
}

func (m *FrozenScorer) forwardEncodedBatch(batch frozenEncodedBatch, shuffle bool) ([][]float32, error) {
	if err := validateFrozenScorer(m, false); err != nil {
		return nil, err
	}
	if shuffle {
		batch = batch.shuffledContexts()
	}
	return m.Head.Forward(batch.Context, batch.ContextMask, batch.Options, batch.OptionMask)
}

func (batch frozenEncodedBatch) shuffledContexts() frozenEncodedBatch {
	if len(batch.Context) <= 1 {
		return batch
	}
	rolled := frozenEncodedBatch{
		Context:     make([][][]float32, len(batch.Context)),
		ContextMask: make([][]bool, len(batch.ContextMask)),
		Options:     batch.Options,
		OptionMask:  batch.OptionMask,
		Labels:      batch.Labels,
	}
	for i := range batch.Context {
		j := (i + len(batch.Context) - 1) % len(batch.Context)
		rolled.Context[i] = batch.Context[j]
		rolled.ContextMask[i] = batch.ContextMask[j]
	}
	return rolled
}

func validateFrozenScorer(m *FrozenScorer, requireEncoder bool) error {
	if m == nil {
		return fmt.Errorf("jevlike frozen scorer is nil")
	}
	if requireEncoder && m.Encoder == nil {
		return fmt.Errorf("missing frozen encoder")
	}
	if err := m.Config.Validate(); err != nil {
		return err
	}
	if m.Head.Width != m.Config.Width || m.Head.Rank != m.Config.Rank {
		return fmt.Errorf("head/config mismatch")
	}
	return m.Head.Validate()
}

func validateEncoded(rows [][]float32, width int) error {
	if len(rows) == 0 {
		return fmt.Errorf("encoder returned no hidden states")
	}
	for _, row := range rows {
		if len(row) != width {
			return fmt.Errorf("encoder hidden width %d, want %d", len(row), width)
		}
	}
	return nil
}
