package jevlike

import (
	"fmt"
	backbone "github.com/rcarmo/go-pherence/model"
)

// TokenEncoder returns final hidden states without an LM-head projection.
// Implementations must encode special tokens and truncate according to their
// own tokenizer contract. They must not mutate model weights.
type TokenEncoder interface {
	Encode(text string, maxTokens int) ([][]float32, error)
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

func (c Checkpoint) Frozen(encoder TokenEncoder) (*FrozenScorer, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	if c.Encoder != "frozen" || encoder == nil {
		return nil, fmt.Errorf("frozen checkpoint and encoder required")
	}
	h, err := NewAttentionHead(c.Config.Width, c.Config.Rank)
	if err != nil {
		return nil, err
	}
	params := make(map[string][]float32)
	for _, p := range c.Parameters {
		params[p.Name] = p.Values
	}
	if err = h.LoadNamedParameters("head", params); err != nil {
		return nil, err
	}
	return &FrozenScorer{Config: c.Config, Reference: c.EncoderReference, Head: *h, Encoder: encoder}, nil
}

func (m *FrozenScorer) Forward(examples []ChoiceExample, shuffle bool) ([][]float32, error) {
	if m == nil || m.Encoder == nil {
		return nil, fmt.Errorf("missing frozen encoder")
	}
	if err := m.Config.Validate(); err != nil {
		return nil, err
	}
	if m.Head.Width != m.Config.Width || m.Head.Rank != m.Config.Rank {
		return nil, fmt.Errorf("head/config mismatch")
	}
	if len(examples) == 0 {
		return nil, fmt.Errorf("empty batch")
	}
	contexts := make([][][]float32, len(examples))
	options := make([][][]float32, len(examples))
	maxCtx, maxOpts := 0, 0
	for i, ex := range examples {
		if _, err := ValidateChoiceExample(ex); err != nil {
			return nil, err
		}
		ctx, err := m.Encoder.Encode(ex.Context, m.Config.ContextTokens)
		if err != nil {
			return nil, err
		}
		if err = validateEncoded(ctx, m.Config.Width); err != nil {
			return nil, err
		}
		contexts[i] = ctx
		maxCtx = max(maxCtx, len(ctx))
		maxOpts = max(maxOpts, len(ex.Options))
		for _, text := range ex.Options {
			tokens, err := m.Encoder.Encode(text, m.Config.OptionTokens)
			if err != nil {
				return nil, err
			}
			if err = validateEncoded(tokens, m.Config.Width); err != nil {
				return nil, err
			}
			pooled := make([]float32, m.Config.Width)
			for _, row := range tokens {
				for d, v := range row {
					pooled[d] += v / float32(len(tokens))
				}
			}
			options[i] = append(options[i], pooled)
		}
	}
	cm := make([][]bool, len(examples))
	om := make([][]bool, len(examples))
	for i := range examples {
		cm[i] = make([]bool, maxCtx)
		for j := range contexts[i] {
			cm[i][j] = true
		}
		for len(contexts[i]) < maxCtx {
			contexts[i] = append(contexts[i], make([]float32, m.Config.Width))
		}
		om[i] = make([]bool, maxOpts)
		for j := range options[i] {
			om[i][j] = true
		}
		for len(options[i]) < maxOpts {
			options[i] = append(options[i], make([]float32, m.Config.Width))
		}
	}
	if shuffle && len(examples) > 1 {
		contexts = append(contexts[len(contexts)-1:], contexts[:len(contexts)-1]...)
		cm = append(cm[len(cm)-1:], cm[:len(cm)-1]...)
	}
	return m.Head.Forward(contexts, cm, options, om)
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
