package mojev

import (
	"fmt"
	"strings"
)

// TextField names one text-only scorer question. Choice rendering is limited
// to the pinned upstream format; other kinds require separate parity fixtures.
type TextField struct {
	Name, Description string
	Options           []string
}

// TextRow is one text-only request with optional per-row candidate menus.
// A nil menu for a field uses that field's declared options.
type TextRow struct {
	State string
	Menus [][]string
}

// TextEncoder supplies bounded text token IDs; model-specific tokenizer
// loading and provenance checks remain the caller's responsibility.
type TextEncoder func(string) ([]int, error)

func renderChoicePrompt(field TextField) (string, error) {
	if field.Name == "" || len(field.Options) < 2 || len(field.Options) > 64 {
		return "", fmt.Errorf("mojev: invalid text field")
	}
	seen := make(map[string]bool, len(field.Options))
	for _, option := range field.Options {
		if option == "" || seen[option] {
			return "", fmt.Errorf("mojev: invalid or duplicate option")
		}
		seen[option] = true
	}
	parts := []string{strings.ReplaceAll(field.Name, "_", " ")}
	if field.Description != "" {
		parts = append(parts, field.Description)
	}
	parts = append(parts, "kind: choice")
	if len(field.Options) <= 8 {
		parts = append(parts, "options: "+strings.Join(field.Options, ", "))
	}
	return strings.Join(parts, " | "), nil
}

// PackTextRows renders pinned choice prompts, encodes each text segment once,
// and delegates owned span packing to PackEncodedRows. It does not insert chat
// templates, process images or execute the encoder. Limits apply per segment.
func PackTextRows(rows []TextRow, fields []TextField, stateLimit, questionLimit, padID int, encode TextEncoder) (*PackedRows, error) {
	if padID < 0 {
		return nil, fmt.Errorf("mojev: invalid text-packing geometry")
	}
	encoded, err := encodeTextRows(rows, fields, stateLimit, questionLimit, encode)
	if err != nil {
		return nil, err
	}
	return PackEncodedRows(encoded, padID)
}

// encodeTextRows keeps validated token segments for branch-local inference;
// public packed callers materialise their masks only after this shared stage.
func encodeTextRows(rows []TextRow, fields []TextField, stateLimit, questionLimit int, encode TextEncoder) ([]EncodedRow, error) {
	if encode == nil || len(fields) == 0 || len(fields) > 256 || len(rows) == 0 || len(rows) > 16 ||
		stateLimit <= 0 || stateLimit > 4096 || questionLimit <= 0 || questionLimit > 4096 {
		return nil, fmt.Errorf("mojev: invalid text-packing geometry")
	}
	prompts := make([][]int, len(fields))
	encodeSegment := func(text string, limit int) ([]int, error) {
		ids, err := encode(text)
		if err != nil {
			return nil, fmt.Errorf("mojev: encode text: %w", err)
		}
		if len(ids) == 0 || len(ids) > 4096 {
			return nil, fmt.Errorf("mojev: empty or oversized encoded segment")
		}
		for _, id := range ids {
			if id < 0 {
				return nil, fmt.Errorf("mojev: negative token ID")
			}
		}
		if limit > 0 && len(ids) > limit {
			ids = ids[:limit]
		}
		return ids, nil
	}
	for f, field := range fields {
		prompt, err := renderChoicePrompt(field)
		if err != nil {
			return nil, err
		}
		prompts[f], err = encodeSegment(prompt, questionLimit)
		if err != nil {
			return nil, err
		}
	}
	encoded := make([]EncodedRow, len(rows))
	for r, row := range rows {
		if row.State == "" || len(row.Menus) != len(fields) {
			return nil, fmt.Errorf("mojev: invalid text row")
		}
		state, err := encodeSegment(row.State, stateLimit)
		if err != nil {
			return nil, err
		}
		encoded[r] = EncodedRow{State: state, Questions: prompts, Candidates: make([][][]int, len(fields))}
		for f, field := range fields {
			menu := row.Menus[f]
			if menu == nil {
				menu = field.Options
			}
			if len(menu) < 2 || len(menu) > len(field.Options) {
				return nil, fmt.Errorf("mojev: invalid per-row menu")
			}
			seen := make(map[string]bool, len(menu))
			encoded[r].Candidates[f] = make([][]int, len(menu))
			for n, candidate := range menu {
				if candidate == "" || seen[candidate] {
					return nil, fmt.Errorf("mojev: invalid or duplicate candidate")
				}
				seen[candidate] = true
				encoded[r].Candidates[f][n], err = encodeSegment(candidate, 0)
				if err != nil {
					return nil, err
				}
			}
		}
		if encodedRowTokenCount(encoded[r]) > 4096 {
			return nil, fmt.Errorf("mojev: packed row exceeds reference limit")
		}
	}
	return encoded, nil
}

func encodedRowTokenCount(row EncodedRow) int {
	n := len(row.State)
	for f, q := range row.Questions {
		n += len(q)
		for _, candidate := range row.Candidates[f] {
			n += len(candidate)
		}
	}
	return n
}
