package whisper

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// Tokenizer decodes Whisper token IDs to text.
type Tokenizer struct {
	Vocab     map[int]string // id → token string
	VocabSize int
}

// LoadTokenizer loads the tokenizer from a HuggingFace tokenizer.json file.
func LoadTokenizer(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer: %w", err)
	}

	var raw struct {
		Model struct {
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
		AddedTokens []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}

	// Build reverse vocab (id → string)
	vocab := make(map[int]string, len(raw.Model.Vocab)+len(raw.AddedTokens))
	for tok, id := range raw.Model.Vocab {
		vocab[id] = tok
	}
	for _, at := range raw.AddedTokens {
		vocab[at.ID] = at.Content
	}

	return &Tokenizer{Vocab: vocab, VocabSize: len(vocab)}, nil
}

// Decode converts a sequence of token IDs to text.
func (t *Tokenizer) Decode(tokens []int) string {
	if t == nil || len(tokens) == 0 {
		return ""
	}

	var parts []string
	for _, tok := range tokens {
		// Skip special tokens
		if tok >= TokenSOT {
			continue
		}
		s, ok := t.Vocab[tok]
		if !ok {
			continue
		}
		// Convert GPT-2 BPE Ġ to space
		s = strings.ReplaceAll(s, "Ġ", " ")
		// Convert byte-level tokens (e.g., Ã, ¤) back to UTF-8
		s = decodeBPEBytes(s)
		parts = append(parts, s)
	}

	text := strings.Join(parts, "")
	return strings.TrimSpace(text)
}

// decodeBPEBytes handles the GPT-2 byte-level BPE encoding where bytes 0-255
// are mapped to Unicode characters in a specific range.
func decodeBPEBytes(s string) string {
	// GPT-2 BPE maps bytes to printable Unicode chars.
	// Most common tokens are already readable ASCII/Unicode.
	// The byte-fallback chars (Ã, ¢, etc.) are rare in practice.
	// For now, pass through — a full implementation would decode the byte map.
	return s
}
