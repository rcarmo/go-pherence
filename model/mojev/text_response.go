package mojev

import (
	"fmt"
	"sort"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

// TextDecision is an owned, model-free response over injected sorted logit rows.
// Usage counts the packed text tokens; no model, processor or HTTP path runs.
type TextDecision struct {
	Model   string
	Answers map[string]map[string]any
	Usage   TextUsage
}

type TextUsage struct {
	InputTokens  int
	OutputTokens int
}

type preparedTextDecision struct {
	req    TextRequest
	fields []TextField
	menus  [][]string
}

func prepareTextDecision(req TextRequest, sortedLogits [][]float64) (*preparedTextDecision, error) {
	if req.Model == "" || req.State == "" || len(req.Fields) == 0 || len(req.Fields) > 256 {
		return nil, fmt.Errorf("mojev: invalid text decision geometry")
	}
	if sortedLogits != nil && len(sortedLogits) != len(req.Fields) {
		return nil, fmt.Errorf("mojev: invalid text decision geometry")
	}
	fields := make([]TextField, len(req.Fields))
	menus := make([][]string, len(req.Fields))
	seen := make(map[string]bool, len(req.Fields))
	for i, field := range req.Fields {
		if field.ID == "" || seen[field.ID] || len(field.Options) != len(field.Keys) || len(field.Options) != len(field.SortedOptions) || len(field.Options) != len(field.SortedIndices) || sortedLogits != nil && len(sortedLogits[i]) != len(field.Options) {
			return nil, fmt.Errorf("mojev: invalid field or logit row %d", i)
		}
		if sortedLogits == nil {
			// Inference must reject unsupported kinds/labels before model work.
			if err := validateAnswerLabels(field.Kind, field.Keys, field.Options); err != nil {
				return nil, err
			}
		}
		seen[field.ID] = true
		// AssembleAnswer independently normalises sorted logits. Require the
		// supplied packing order to be the same stable text sort it uses.
		order := make([]int, len(field.Options))
		for j := range order {
			order[j] = j
		}
		sort.SliceStable(order, func(a, b int) bool { return field.Options[order[a]] < field.Options[order[b]] })
		for j, original := range order {
			if field.SortedIndices[j] != original || field.SortedOptions[j] != field.Options[original] {
				return nil, fmt.Errorf("mojev: invalid sorted candidate order")
			}
		}
		fields[i] = TextField{Name: field.ID, Description: field.Instructions, Options: field.SortedOptions}
		menus[i] = field.SortedOptions
	}
	return &preparedTextDecision{req: req, fields: fields, menus: menus}, nil
}

func (p *preparedTextDecision) pack(stateLimit, questionLimit, padID int, encode TextEncoder) (*PackedRows, error) {
	if p == nil {
		return nil, fmt.Errorf("mojev: invalid text decision geometry")
	}
	return PackTextRows([]TextRow{{State: p.req.State, Menus: p.menus}}, p.fields, stateLimit, questionLimit, padID, encode)
}

func (p *preparedTextDecision) assemble(sortedLogits [][]float64, packedMask []bool) (*TextDecision, error) {
	if p == nil || len(sortedLogits) != len(p.req.Fields) {
		return nil, fmt.Errorf("mojev: invalid text decision geometry")
	}
	out := &TextDecision{Model: p.req.Model, Answers: make(map[string]map[string]any, len(p.req.Fields))}
	for i, field := range p.req.Fields {
		if len(sortedLogits[i]) != len(field.Options) {
			return nil, fmt.Errorf("mojev: invalid field or logit row %d", i)
		}
		answer, err := AssembleAnswer(field.Kind, field.Keys, field.Options, sortedLogits[i])
		if err != nil {
			return nil, fmt.Errorf("mojev: answer %q: %w", field.ID, err)
		}
		out.Answers[field.ID] = answer
	}
	for _, present := range packedMask {
		if present {
			out.Usage.InputTokens++
		}
	}
	return out, nil
}

// AssembleTextDecision combines a decoded text request, an injected text
// encoder and one sorted logit row per question. It validates every field and
// returns no partial answers if packing or any row fails. Callers using the
// released tokenizer must first apply ValidateTextControls; this lower-level
// injected-encoder API cannot inspect its tokenizer policy.
func AssembleTextDecision(req TextRequest, sortedLogits [][]float64, encode TextEncoder, padID, stateLimit, questionLimit int) (*TextDecision, error) {
	if len(sortedLogits) != len(req.Fields) {
		return nil, fmt.Errorf("mojev: invalid text decision geometry")
	}
	prepared, err := prepareTextDecision(req, sortedLogits)
	if err != nil {
		return nil, err
	}
	packed, err := prepared.pack(stateLimit, questionLimit, padID, encode)
	if err != nil {
		return nil, err
	}
	return prepared.assemble(sortedLogits, packed.PackedMask[0])
}

// AssembleSafeTextDecision validates reserved tokens with the released
// tokenizer before passing its Encode method to the model-free assembler.
// It returns no partial response if validation fails.
func AssembleSafeTextDecision(req TextRequest, sortedLogits [][]float64, tok *tokenizer.Tokenizer, padID, stateLimit, questionLimit int) (*TextDecision, error) {
	if err := ValidateTextControls(req, tok); err != nil {
		return nil, err
	}
	return AssembleTextDecision(req, sortedLogits, func(text string) ([]int, error) {
		return tok.Encode(text), nil
	}, padID, stateLimit, questionLimit)
}
