package simplejev

import (
	"fmt"
	"math"
	"slices"
)

const MaxPromptBytes = 64 << 10
const MaxPromptTokens = 8192
const MaxModelLabelBytes = 64

// RenderedLabel binds one public ID to one model-facing label string.
type RenderedLabel struct {
	PublicID  string
	ModelText string
}

// PromptRenderer supplies an independently chosen model-facing prefix and
// explicit public-ID mappings. It must not return a completed answer.
type PromptRenderer interface {
	Render(state string, question Question) (prefix string, labels []RenderedLabel, err error)
}

// PromptTokenizer encodes a complete prompt. The adapter compares encoding of
// prefix and prefix+label to verify a single token at the actual boundary.
type PromptTokenizer interface {
	Encode(text string) ([]int, error)
}

// SelectedLogitBackend scores only admitted token IDs after one prompt. It
// must return one finite logit per token in input order, without generating.
type SelectedLogitBackend interface {
	SelectedLogits(prompt []int, tokens []int) ([]float32, error)
}

// TokenLogitAdapter is a model-free LogitProvider. The three injected parts
// own prompt rendering, tokenisation and inference; no upstream template is
// shipped here. One backend call is made per validated question.
type TokenLogitAdapter struct {
	Renderer  PromptRenderer
	Tokenizer PromptTokenizer
	Backend   SelectedLogitBackend
}

func (a TokenLogitAdapter) LabelLogits(state string, question Question) ([]Label, error) {
	if a.Renderer == nil || a.Tokenizer == nil || a.Backend == nil {
		return nil, fmt.Errorf("simplejev: incomplete token-logit adapter")
	}
	if err := (Request{State: state, Questions: []Question{question}}).Validate(); err != nil {
		return nil, err
	}
	// The renderer receives its own labels, not the caller's request storage.
	prepared := question
	prepared.Labels = slices.Clone(question.Labels)
	prefix, names, err := a.Renderer.Render(state, prepared)
	if err != nil {
		return nil, err
	}
	if len(prefix) == 0 || len(prefix) > MaxPromptBytes || len(names) != len(question.Labels) {
		return nil, fmt.Errorf("simplejev: invalid rendered prompt or label count")
	}
	prompt, err := a.Tokenizer.Encode(prefix)
	if err != nil {
		return nil, err
	}
	if len(prompt) == 0 || len(prompt) > MaxPromptTokens {
		return nil, fmt.Errorf("simplejev: invalid prompt token count")
	}
	// Some tokenizers reuse one internal buffer; freeze the validated prefix
	// before encoding any prefix+label strings.
	prompt = slices.Clone(prompt)
	for _, id := range prompt {
		if id < 0 {
			return nil, fmt.Errorf("simplejev: invalid prompt token")
		}
	}
	tokens := make([]int, len(names))
	seen := make(map[int]bool, len(names))
	for i, name := range names {
		if name.PublicID != question.Labels[i] || len(name.ModelText) == 0 || len(name.ModelText) > MaxModelLabelBytes || len(prefix)+len(name.ModelText) > MaxPromptBytes {
			return nil, fmt.Errorf("simplejev: invalid model label mapping")
		}
		joined, err := a.Tokenizer.Encode(prefix + name.ModelText)
		if err != nil {
			return nil, err
		}
		if len(joined) != len(prompt)+1 || !slices.Equal(joined[:len(prompt)], prompt) || joined[len(prompt)] < 0 || seen[joined[len(prompt)]] {
			return nil, fmt.Errorf("simplejev: model labels must add distinct single tokens at the prompt boundary")
		}
		tokens[i] = joined[len(prompt)]
		seen[tokens[i]] = true
	}
	// The backend may reuse/mutate its input buffers. Preserve the validated
	// token identity and prompt until its output has been checked.
	backendTokens := slices.Clone(tokens)
	logits, err := a.Backend.SelectedLogits(slices.Clone(prompt), backendTokens)
	if err != nil {
		return nil, err
	}
	if len(logits) != len(tokens) {
		return nil, fmt.Errorf("simplejev: selected logit count mismatch")
	}
	labels := make([]Label, len(tokens))
	for i, value := range logits {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("simplejev: non-finite selected logit")
		}
		labels[i] = Label{ID: question.Labels[i], TokenID: tokens[i], Logit: value}
	}
	return labels, nil
}
