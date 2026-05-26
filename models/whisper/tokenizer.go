package whisper

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode/utf8"
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

	var raw strings.Builder
	for _, tok := range tokens {
		// Skip special tokens
		if tok >= TokenSOT {
			continue
		}
		s, ok := t.Vocab[tok]
		if !ok {
			continue
		}
		raw.WriteString(s)
	}

	text := decodeBPEBytes(raw.String())
	return strings.TrimSpace(text)
}

var byteDecoder = buildByteDecoder()

// decodeBPEBytes handles the GPT-2/Whisper byte-level BPE encoding where bytes
// 0-255 are mapped to printable Unicode characters. This must run over the
// concatenated token string so multi-byte UTF-8 sequences split across tokens
// (for example "TelefÃ³nica") are reconstructed correctly.
func decodeBPEBytes(s string) string {
	if s == "" {
		return ""
	}
	buf := make([]byte, 0, len(s))
	for _, r := range s {
		if b, ok := byteDecoder[r]; ok {
			buf = append(buf, b)
			continue
		}
		var tmp [utf8.UTFMax]byte
		n := utf8.EncodeRune(tmp[:], r)
		buf = append(buf, tmp[:n]...)
	}
	return string(buf)
}

func buildByteDecoder() map[rune]byte {
	bs := make([]int, 0, 256)
	for b := int('!'); b <= int('~'); b++ {
		bs = append(bs, b)
	}
	for b := 0xA1; b <= 0xAC; b++ {
		bs = append(bs, b)
	}
	for b := 0xAE; b <= 0xFF; b++ {
		bs = append(bs, b)
	}
	seen := make(map[int]bool, len(bs))
	for _, b := range bs {
		seen[b] = true
	}
	cs := append([]int(nil), bs...)
	n := 0
	for b := 0; b < 256; b++ {
		if !seen[b] {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}
	dec := make(map[rune]byte, 256)
	for i, b := range bs {
		dec[rune(cs[i])] = byte(b)
	}
	return dec
}
