package qev

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

type rawSchema struct {
	Type       string                     `json:"type"`
	Properties map[string]json.RawMessage `json:"properties"`
}

type rawField struct {
	Type        string            `json:"type"`
	Description string            `json:"description"`
	Choices     []json.RawMessage `json:"choices"`
	Enum        []json.RawMessage `json:"enum"`
}

// CompileSchema accepts compact field objects and JSON Schema properties.
// QEV v1 deliberately supports only boolean and string enum fields.
func CompileSchema(data []byte, instructions string) (CompiledSchema, error) {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return CompiledSchema{}, fmt.Errorf("schema must be an object")
	}
	dec := json.NewDecoder(bytes.NewReader(trimmed))
	dec.UseNumber()
	var top map[string]json.RawMessage
	if err := dec.Decode(&top); err != nil {
		return CompiledSchema{}, fmt.Errorf("schema: %w", err)
	}
	if top == nil {
		return CompiledSchema{}, fmt.Errorf("schema must be an object")
	}
	props := top
	jsonSchema := false
	if raw, ok := top["properties"]; ok {
		jsonSchema = true
		var decoded map[string]json.RawMessage
		if err := json.Unmarshal(raw, &decoded); err != nil || decoded == nil {
			return CompiledSchema{}, fmt.Errorf("schema properties must be an object")
		}
		props = decoded
	}
	if len(props) < 1 || len(props) > MaxFields {
		return CompiledSchema{}, fmt.Errorf("schema must define 1-%d fields", MaxFields)
	}

	// encoding/json maps do not preserve source order. Sort field names so IDs,
	// prompt text and responses remain stable across processes and platforms.
	names := make([]string, 0, len(props))
	for name := range props {
		names = append(names, name)
	}
	sort.Strings(names)

	out := CompiledSchema{Fields: make([]FieldSpec, 0, len(names)), Inputs: make([]FieldInput, 0, len(names))}
	var catalog strings.Builder
	for _, name := range names {
		if name == "" {
			return CompiledSchema{}, fmt.Errorf("field name must not be empty")
		}
		var raw rawField
		if err := json.Unmarshal(props[name], &raw); err != nil {
			return CompiledSchema{}, fmt.Errorf("field %q must be an object", name)
		}
		if !jsonSchema && strings.TrimSpace(raw.Description) == "" {
			return CompiledSchema{}, fmt.Errorf("field %q needs a description", name)
		}
		field, err := compileField(name, raw)
		if err != nil {
			return CompiledSchema{}, err
		}
		common := commonPrefix(field.Candidates)
		input := FieldInput{Suffix: "  " + strconv.Quote(name) + ": " + common, Candidates: make([]Candidate, len(field.Candidates))}
		for i, candidate := range field.Candidates {
			input.Candidates[i] = candidate
			input.Candidates[i].Encoded = strings.TrimPrefix(candidate.Encoded, common)
		}
		if catalog.Len() > 0 {
			catalog.WriteByte('\n')
		}
		catalog.WriteString(strconv.Quote(name))
		if field.Description != "" {
			catalog.WriteString(": ")
			catalog.WriteString(field.Description)
		}
		catalog.WriteString("\nAllowed values: ")
		for i, candidate := range field.Candidates {
			if i > 0 {
				catalog.WriteString(", ")
			}
			catalog.WriteString(candidate.Encoded)
		}
		out.Fields = append(out.Fields, field)
		out.Inputs = append(out.Inputs, input)
	}
	out.SystemText = "Select the requested field value from its allowed values, based on the context. Respond with the JSON value only.\n\nFields:\n" + catalog.String()
	if instructions != "" {
		out.SystemText += "\n" + instructions
	}
	return out, nil
}

func compileField(name string, raw rawField) (FieldSpec, error) {
	field := FieldSpec{Name: name, Type: raw.Type, Description: raw.Description}
	values := raw.Choices
	if len(raw.Enum) > 0 {
		values = raw.Enum
		field.Type = "enum"
	}
	switch field.Type {
	case "boolean":
		values = []json.RawMessage{json.RawMessage("true"), json.RawMessage("false")}
	case "enum", "choice", "selection", "string":
		field.Type = "enum"
		if len(values) == 0 {
			return FieldSpec{}, fmt.Errorf("field %q: enum fields need choices", name)
		}
	default:
		return FieldSpec{}, fmt.Errorf("field %q: supported types are boolean and enum", name)
	}
	if len(values) > MaxCandidates {
		return FieldSpec{}, fmt.Errorf("field %q needs 1-%d allowed values", name, MaxCandidates)
	}
	seen := make(map[string]struct{}, len(values))
	field.Candidates = make([]Candidate, 0, len(values))
	for i, value := range values {
		var typed any
		dec := json.NewDecoder(bytes.NewReader(value))
		dec.UseNumber()
		if err := dec.Decode(&typed); err != nil {
			return FieldSpec{}, fmt.Errorf("field %q candidate %d: %w", name, i, err)
		}
		if field.Type == "enum" {
			if _, ok := typed.(string); !ok {
				return FieldSpec{}, fmt.Errorf("field %q: enum choices must be strings", name)
			}
		}
		encoded, err := json.Marshal(typed)
		if err != nil {
			return FieldSpec{}, fmt.Errorf("field %q candidate %d: %w", name, i, err)
		}
		key := string(encoded)
		if _, ok := seen[key]; ok {
			return FieldSpec{}, fmt.Errorf("field %q has duplicate allowed value %s", name, key)
		}
		seen[key] = struct{}{}
		field.Candidates = append(field.Candidates, Candidate{
			ID:      fmt.Sprintf("%s:%03d", name, i),
			Encoded: key,
			Value:   append(json.RawMessage(nil), encoded...),
		})
	}
	return field, nil
}

func commonPrefix(candidates []Candidate) string {
	if len(candidates) == 0 {
		return ""
	}
	prefix := candidates[0].Encoded
	for _, candidate := range candidates[1:] {
		n := 0
		for n < len(prefix) && n < len(candidate.Encoded) && prefix[n] == candidate.Encoded[n] {
			n++
		}
		prefix = prefix[:n]
	}
	return prefix
}
