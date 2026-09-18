package gliner2

import (
	"fmt"
	"strings"
)

// Ordered descriptions and examples preserve the upstream prompt order.
// Map iteration must never determine the token sequence.
type LabelDescription struct {
	Label string `json:"label"`
	Text  string `json:"text"`
}
type SchemaExample struct {
	Input string `json:"input"`
	Label string `json:"label"`
}
type TextSchema struct {
	Parent       string             `json:"parent"`
	Marker       string             `json:"marker"`
	Labels       []string           `json:"labels"`
	Prompt       string             `json:"prompt,omitempty"`
	Descriptions []LabelDescription `json:"descriptions,omitempty"`
	Examples     []SchemaExample    `json:"examples,omitempty"`
	Record       *RecordSpec        `json:"record,omitempty"`
}

// PrepareTextSchema matches processor._transform_schema's descriptions/both
// construction. Only structural marker slots become queries; literal special
// tokens inside prompt text, labels or examples do not add query positions.
func (t *Tokenizer) PrepareTextSchema(text string, s TextSchema, maxTokens int) (EntityInput, error) {
	if strings.TrimSpace(s.Parent) == "" {
		return EntityInput{}, fmt.Errorf("schema parent required")
	}
	switch s.Marker {
	case "[E]", "[L]", "[R]", "[C]":
	default:
		return EntityInput{}, fmt.Errorf("unsupported schema marker %q", s.Marker)
	}
	if err := validateTextSchemaRecordMetadata(s); err != nil {
		return EntityInput{}, err
	}
	allowed := make(map[string]bool, len(s.Labels))
	for _, label := range s.Labels {
		allowed[label] = true
	}
	prompt := s.Parent
	if s.Prompt != "" {
		prompt += ": " + s.Prompt
	}
	for _, d := range s.Descriptions {
		if allowed[d.Label] {
			prompt += " [DESCRIPTION] " + d.Label + ": " + d.Text
		}
	}
	for _, e := range s.Examples {
		if allowed[e.Label] {
			prompt += " [EXAMPLE] " + e.Input + " [OUTPUT] " + e.Label
		}
	}
	return t.prepareSchema(text, prompt, s.Marker, s.Labels, maxTokens)
}
