// Package gosystemone implements finite-schema decisions over Gemma4 token logits.
package gosystemone

import (
	"encoding/json"
	"fmt"
)

const (
	MaxContexts       = 256
	MaxFields         = 32
	MaxCandidates     = 255
	MaxContextTokens  = 32 * 1024
	DefaultTreeMax    = 128
	DecisionSequences = 12
	UBatchTokens      = 512
)

// Mode controls how finite candidate paths are scored.
type Mode string

const (
	ModeAuto Mode = "auto"
	ModeTree Mode = "tree"
)

// Request is the JSON contract accepted by POST /v1/decision.
type Request struct {
	Model        string          `json:"model,omitempty"`
	Instructions string          `json:"instructions,omitempty"`
	Schema       json.RawMessage `json:"schema"`
	Contexts     []string        `json:"contexts"`
	Mode         Mode            `json:"mode,omitempty"`
	TreeMax      int             `json:"tree_max,omitempty"`
	CachePrompt  *bool           `json:"cache_prompt,omitempty"`
}

func (r *Request) NormalizeAndValidate() error {
	if r == nil {
		return fmt.Errorf("nil decision request")
	}
	if len(r.Contexts) < 1 || len(r.Contexts) > MaxContexts {
		return fmt.Errorf("contexts must contain 1-%d strings", MaxContexts)
	}
	for i, context := range r.Contexts {
		if context == "" {
			return fmt.Errorf("contexts[%d] must not be empty", i)
		}
	}
	if len(r.Schema) == 0 {
		return fmt.Errorf("schema is required")
	}
	if r.Mode == "" {
		r.Mode = ModeAuto
	}
	if r.Mode != ModeAuto && r.Mode != ModeTree {
		return fmt.Errorf("mode must be auto or tree")
	}
	if r.TreeMax == 0 {
		r.TreeMax = DefaultTreeMax
	}
	if r.TreeMax < 1 || r.TreeMax > MaxCandidates {
		return fmt.Errorf("tree_max must be 1-%d", MaxCandidates)
	}
	return nil
}

func (r Request) AllowCache() bool {
	return r.CachePrompt == nil || *r.CachePrompt
}

// Candidate has a stable request-local ID, its exact JSON representation and
// the typed value copied into the response.
type Candidate struct {
	ID      string
	Encoded string
	Value   json.RawMessage
}

type FieldSpec struct {
	Name        string
	Type        string
	Description string
	Candidates  []Candidate
}

type FieldInput struct {
	Suffix     string
	Candidates []Candidate
}

type CompiledSchema struct {
	SystemText string
	Fields     []FieldSpec
	Inputs     []FieldInput
}

type FieldResult struct {
	Value       json.RawMessage `json:"value"`
	Probability float64         `json:"probability"`
	ScoredNodes int             `json:"scored_nodes"`
	Tree        bool            `json:"tree"`
}

type ItemUsage struct {
	ContextTokens int `json:"context_tokens"`
	ScoredRows    int `json:"scored_rows"`
}

type Result struct {
	Decision map[string]json.RawMessage `json:"decision"`
	Fields   map[string]FieldResult     `json:"fields"`
	Usage    ItemUsage                  `json:"usage"`
}

type Usage struct {
	PromptTokens  int `json:"prompt_tokens"`
	CachedTokens  int `json:"cached_tokens"`
	ContextTokens int `json:"context_tokens"`
	ScoredRows    int `json:"scored_rows"`
}

type Timings struct {
	PrefillMS     float64 `json:"prefill_ms"`
	ScoringMS     float64 `json:"scoring_ms"`
	TotalMS       float64 `json:"total_ms"`
	Rounds        int     `json:"rounds"`
	PerDecisionMS float64 `json:"per_decision_ms"`
}

type Response struct {
	Object  string   `json:"object"`
	Model   string   `json:"model"`
	Created int64    `json:"created"`
	Results []Result `json:"results"`
	Usage   Usage    `json:"usage"`
	Timings Timings  `json:"timings"`
}
