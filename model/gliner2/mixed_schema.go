package gliner2

import "fmt"

// SchemaGroupRouting maps only structural choice-marker positions. Parent
// markers and literal markers in descriptions never become extra queries.
type SchemaGroupRouting struct {
	Schema         TextSchema
	QueryPositions []int
}
type MixedInput struct {
	IDs           []int
	Words         []Word
	TextPositions []int
	Groups        []SchemaGroupRouting
}

// PrepareSchemas formats all groups before a single shared text segment.
// Per-group preparation is used only for token routing, never model inference.
func (t *Tokenizer) PrepareSchemas(text string, schemas []TextSchema, maxTokens int) (MixedInput, error) {
	if len(schemas) == 0 || maxTokens <= 0 {
		return MixedInput{}, fmt.Errorf("schemas and positive token budget required")
	}
	sep, err := t.Encode("[SEP_TEXT]", false)
	if err != nil || len(sep) != 1 {
		return MixedInput{}, fmt.Errorf("tokenizer requires atomic [SEP_TEXT]")
	}
	between, err := t.Encode("[SEP_STRUCT]", false)
	if err != nil || len(between) != 1 {
		return MixedInput{}, fmt.Errorf("tokenizer requires atomic [SEP_STRUCT]")
	}
	words, err := SplitWords(text)
	if err != nil {
		return MixedInput{}, err
	}
	out := MixedInput{Words: words}
	for i, s := range schemas {
		part, err := t.PrepareTextSchema("", s, maxTokens)
		if err != nil {
			return MixedInput{}, err
		}
		if len(part.IDs) == 0 || part.IDs[len(part.IDs)-1] != sep[0] {
			return MixedInput{}, fmt.Errorf("schema terminator mismatch")
		}
		if i > 0 {
			out.IDs = append(out.IDs, between[0])
		}
		offset := len(out.IDs)
		group := SchemaGroupRouting{Schema: s, QueryPositions: make([]int, len(part.QueryPositions))}
		for j, p := range part.QueryPositions {
			group.QueryPositions[j] = offset + p
		}
		out.Groups = append(out.Groups, group)
		out.IDs = append(out.IDs, part.IDs[:len(part.IDs)-1]...)
	}
	out.IDs = append(out.IDs, sep[0])
	for _, word := range words {
		ids, err := t.Encode(word.Text, false)
		if err != nil {
			return MixedInput{}, err
		}
		if len(ids) == 0 {
			return MixedInput{}, fmt.Errorf("empty word tokenization")
		}
		out.TextPositions = append(out.TextPositions, len(out.IDs))
		out.IDs = append(out.IDs, ids...)
	}
	if len(out.IDs) > maxTokens {
		return MixedInput{}, fmt.Errorf("combined schema/text length %d exceeds %d", len(out.IDs), maxTokens)
	}
	return out, nil
}
