// Package gguf tokenizer: encode via llama-tokenize, decode via GGUF vocab.
package gguf

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

// Tokenizer wraps the GGUF vocabulary for encode/decode.
type Tokenizer struct {
	modelPath string
	vocab     []string // index → token string (SentencePiece format)
	bosID     int
	eosID     int
}

// NewTokenizer builds a Tokenizer from an open GGUF.
// g must still be open (vocab is read from g.Meta).
func NewTokenizer(g *GGUF) (*Tokenizer, error) {
	raw, ok := g.Meta["tokenizer.ggml.tokens"]
	if !ok {
		return nil, fmt.Errorf("gguf tokenizer: no tokenizer.ggml.tokens in metadata")
	}
	arr, ok := raw.([]any)
	if !ok {
		return nil, fmt.Errorf("gguf tokenizer: tokens not an array")
	}
	vocab := make([]string, len(arr))
	for i, v := range arr {
		s, ok := v.(string)
		if !ok {
			vocab[i] = fmt.Sprintf("<unk%d>", i)
		} else {
			vocab[i] = s
		}
	}
	bos := 1
	eos := 2
	if v, ok := g.MetaUint32("tokenizer.ggml.bos_token_id"); ok {
		bos = int(v)
	}
	if v, ok := g.MetaUint32("tokenizer.ggml.eos_token_id"); ok {
		eos = int(v)
	}
	return &Tokenizer{vocab: vocab, bosID: bos, eosID: eos}, nil
}

// SetModelPath sets the GGUF model path used by llama-tokenize for encoding.
func (t *Tokenizer) SetModelPath(path string) { t.modelPath = path }

// BOS returns the beginning-of-sequence token ID.
func (t *Tokenizer) BOS() int { return t.bosID }

// EOS returns the end-of-sequence token ID.
func (t *Tokenizer) EOS() int { return t.eosID }

// VocabSize returns the vocabulary size.
func (t *Tokenizer) VocabSize() int { return len(t.vocab) }

// Encode tokenizes text by calling llama-tokenize. Requires modelPath to be set.
// It prepends the BOS token automatically (matching llama.cpp behaviour for LLaMA SPM).
func (t *Tokenizer) Encode(text string) ([]int, error) {
	if t.modelPath == "" {
		return nil, fmt.Errorf("gguf tokenizer: modelPath not set; call SetModelPath first")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "llama-tokenize", "--model", t.modelPath, "-p", text)
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &bytes.Buffer{} // suppress startup noise
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("llama-tokenize: %w", err)
	}
	var ids []int
	for _, line := range strings.Split(out.String(), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		// Expected format: "12345 -> 'token text'"
		parts := strings.SplitN(line, " ->", 2)
		if len(parts) < 1 {
			continue
		}
		id, err := strconv.Atoi(strings.TrimSpace(parts[0]))
		if err != nil {
			continue
		}
		ids = append(ids, id)
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("llama-tokenize: no tokens parsed from output: %s", out.String())
	}
	return ids, nil
}

// Decode converts token IDs back to a UTF-8 string.
// SentencePiece tokens use '▁' (U+2581) for space; we map it to ASCII space.
func (t *Tokenizer) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(t.vocab) {
			continue
		}
		tok := t.vocab[id]
		// Map SentencePiece leading space marker to ASCII space.
		tok = strings.ReplaceAll(tok, "\u2581", " ")
		// Byte tokens: single-byte tokens encoded as <0xNN>
		if len(tok) == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && tok[5] == '>' {
			b, err := strconv.ParseUint(tok[3:5], 16, 8)
			if err == nil {
				sb.WriteByte(byte(b))
				continue
			}
		}
		// Emit only valid UTF-8 sequences; drop replacement characters.
		for len(tok) > 0 {
			r, size := utf8.DecodeRuneInString(tok)
			if r != utf8.RuneError {
				sb.WriteRune(r)
			}
			tok = tok[size:]
		}
	}
	return sb.String()
}
