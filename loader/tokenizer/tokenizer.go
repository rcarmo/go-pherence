package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"
	"sync"
	"unicode"
)

// Tokenizer handles BPE tokenization for LLaMA-style models.
type Tokenizer struct {
	Vocab    map[string]int // token string → ID
	InvVocab map[int]string // ID → token string
	Merges   [][2]string    // BPE merge pairs in priority order
}

// Load loads a HuggingFace tokenizer.json.
func Load(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var raw struct {
		Model struct {
			Vocab  map[string]int  `json:"vocab"`
			Merges json.RawMessage `json:"merges"`
		} `json:"model"`
		AddedTokens []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}

	if raw.Model.Vocab == nil {
		raw.Model.Vocab = map[string]int{}
	}
	t := &Tokenizer{
		Vocab:    raw.Model.Vocab,
		InvVocab: make(map[int]string, len(raw.Model.Vocab)),
	}
	for k, v := range raw.Model.Vocab {
		t.InvVocab[v] = k
	}
	// Add special/added tokens
	for _, at := range raw.AddedTokens {
		if _, exists := t.Vocab[at.Content]; !exists {
			t.Vocab[at.Content] = at.ID
		}
		t.InvVocab[at.ID] = at.Content
	}

	if len(raw.Model.Merges) == 0 || string(raw.Model.Merges) == "null" {
		return t, nil
	}

	// Merges can be ["a b", ...] (strings) or [["a","b"], ...] (arrays)
	var mergeStrings []string
	if err := json.Unmarshal(raw.Model.Merges, &mergeStrings); err == nil {
		t.Merges = make([][2]string, 0, len(mergeStrings))
		for i, m := range mergeStrings {
			parts := strings.SplitN(m, " ", 2)
			if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
				return nil, fmt.Errorf("malformed merge at index %d", i)
			}
			t.Merges = append(t.Merges, [2]string{parts[0], parts[1]})
		}
	} else {
		var mergeArrays [][2]string
		if err := json.Unmarshal(raw.Model.Merges, &mergeArrays); err == nil {
			t.Merges = make([][2]string, 0, len(mergeArrays))
			for i, m := range mergeArrays {
				if m[0] == "" || m[1] == "" {
					return nil, fmt.Errorf("malformed merge at index %d", i)
				}
				t.Merges = append(t.Merges, m)
			}
		} else {
			return nil, fmt.Errorf("unsupported merges format")
		}
	}

	return t, nil
}

// Encode tokenizes a string into token IDs.
func (t *Tokenizer) Encode(text string) []int {
	if t == nil || t.Vocab == nil {
		return nil
	}
	// Auto-detect family: Ġ (U+0120, GPT-2/Qwen byte-level BPE) or ▁
	// (U+2581, SentencePiece/Gemma). SentencePiece keeps the legacy
	// whitespace-prefix path; GPT-2/Qwen uses faithful byte-level BPE.
	if _, ok := t.Vocab["\u2581the"]; ok {
		return t.encodeSentencePiece(text)
	}
	return t.encodeByteLevel(text)
}

// gpt2Pattern is the Qwen2/Qwen3 byte-level pre-tokenization regex. RE2 has no
// lookahead, so the trailing-whitespace `\s+(?!\S)` clause is dropped and its
// effect (handing the final space of an interior whitespace run to the
// following token) is reproduced in splitWhitespaceRuns.
var gpt2Pattern = regexp.MustCompile(`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+`)

// splitWhitespaceRuns emulates the `\s+(?!\S)` lookahead: for an interior
// whitespace run that ends in a space and is followed by another token, the
// trailing space is moved to the front of that next token (matching the
// leading-space handling of the letter/number/symbol classes).
func splitWhitespaceRuns(pieces []string) []string {
	out := make([]string, 0, len(pieces))
	for i := 0; i < len(pieces); i++ {
		p := pieces[i]
		if i+1 < len(pieces) && len(p) >= 2 && isSpaceRun(p) {
			last := p[len(p)-1]
			// A trailing space is accepted as a leading char by every
			// class; a trailing tab only by the letter/number classes.
			if last == ' ' || (last == '\t' && startsAlnum(pieces[i+1])) {
				out = append(out, p[:len(p)-1])
				pieces[i+1] = string(last) + pieces[i+1]
				continue
			}
		}
		out = append(out, p)
	}
	return out
}

func startsAlnum(s string) bool {
	for _, r := range s {
		return unicode.IsLetter(r) || unicode.IsNumber(r)
	}
	return false
}

func isSpaceRun(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] != ' ' && s[i] != '\t' {
			return false
		}
	}
	return len(s) > 0
}

// encodeByteLevel performs faithful Qwen byte-level BPE: pre-tokenization,
// per-byte unicode mapping, then rank-ordered merges over the byte symbols
// (matching the inverse applied by Decode).
func (t *Tokenizer) encodeByteLevel(text string) []int {
	mergeRank := make(map[[2]string]int, len(t.Merges))
	for i, m := range t.Merges {
		mergeRank[m] = i
	}
	byteEncoder := getByteEncoder()

	pieces := splitWhitespaceRuns(gpt2Pattern.FindAllString(text, -1))
	var ids []int
	for _, piece := range pieces {
		// Map each raw UTF-8 byte (not rune) through the GPT-2 byte encoder.
		symbols := make([]string, 0, len(piece))
		for i := 0; i < len(piece); i++ {
			symbols = append(symbols, string(byteEncoder[piece[i]]))
		}
		if len(symbols) == 0 {
			continue
		}
		ids = append(ids, t.bpeMerge(symbols, mergeRank)...)
	}
	return ids
}

// bpeMerge applies rank-ordered pair merges to a symbol list and resolves the
// result to vocab IDs.
func (t *Tokenizer) bpeMerge(symbols []string, mergeRank map[[2]string]int) []int {
	// Direct lookup for the whole joined piece first.
	if joined := strings.Join(symbols, ""); len(symbols) > 1 {
		if id, ok := t.Vocab[joined]; ok {
			return []int{id}
		}
	}
	for len(symbols) >= 2 {
		bestRank := len(t.Merges)
		bestIdx := -1
		for i := 0; i < len(symbols)-1; i++ {
			if rank, ok := mergeRank[[2]string{symbols[i], symbols[i+1]}]; ok && rank < bestRank {
				bestRank = rank
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break
		}
		merged := symbols[bestIdx] + symbols[bestIdx+1]
		newSyms := make([]string, 0, len(symbols)-1)
		newSyms = append(newSyms, symbols[:bestIdx]...)
		newSyms = append(newSyms, merged)
		newSyms = append(newSyms, symbols[bestIdx+2:]...)
		symbols = newSyms
	}
	ids := make([]int, 0, len(symbols))
	for _, s := range symbols {
		if id, ok := t.Vocab[s]; ok {
			ids = append(ids, id)
		}
	}
	return ids
}

// encodeSentencePiece is the legacy whitespace-prefix path used for
// SentencePiece-family vocabularies (Gemma).
func (t *Tokenizer) encodeSentencePiece(text string) []int {
	spacePrefix := "\u2581"
	words := strings.Fields(text)
	var pieces []string
	for i, w := range words {
		if i > 0 {
			w = spacePrefix + w
		}
		pieces = append(pieces, w)
	}

	// For each piece, try direct vocab lookup first, then BPE
	mergeRank := make(map[[2]string]int, len(t.Merges))
	for i, m := range t.Merges {
		mergeRank[m] = i
	}

	var ids []int
	for _, piece := range pieces {
		// Direct lookup
		if id, ok := t.Vocab[piece]; ok {
			ids = append(ids, id)
			continue
		}

		// BPE: split into characters
		chars := make([]string, 0, len(piece))
		for _, r := range piece {
			chars = append(chars, string(r))
		}

		// Apply BPE merges
		for len(chars) >= 2 {
			bestRank := len(t.Merges)
			bestIdx := -1
			for i := 0; i < len(chars)-1; i++ {
				pair := [2]string{chars[i], chars[i+1]}
				if rank, ok := mergeRank[pair]; ok && rank < bestRank {
					bestRank = rank
					bestIdx = i
				}
			}
			if bestIdx < 0 {
				break
			}
			merged := chars[bestIdx] + chars[bestIdx+1]
			newChars := make([]string, 0, len(chars)-1)
			newChars = append(newChars, chars[:bestIdx]...)
			newChars = append(newChars, merged)
			newChars = append(newChars, chars[bestIdx+2:]...)
			chars = newChars
		}

		for _, ch := range chars {
			if id, ok := t.Vocab[ch]; ok {
				ids = append(ids, id)
			}
		}
	}
	return ids
}

// Decode converts token IDs back to text.
func (t *Tokenizer) Decode(ids []int) string {
	if t == nil || t.InvVocab == nil {
		return ""
	}
	var parts []string
	for _, id := range ids {
		if tok, ok := t.InvVocab[id]; ok {
			parts = append(parts, tok)
		}
	}
	text := strings.Join(parts, "")
	// Replace SentencePiece space marker with actual space
	text = strings.ReplaceAll(text, "\u2581", " ")
	// Reverse byte-level BPE encoding
	byteDecoder := getByteDecoder()
	var decoded []byte
	for _, r := range text {
		if b, ok := byteDecoder[r]; ok {
			decoded = append(decoded, b)
		} else {
			decoded = append(decoded, string(r)...)
		}
	}
	text = string(decoded)
	return text
}

// VocabSize returns the vocabulary size.
func (t *Tokenizer) VocabSize() int {
	if t == nil || t.Vocab == nil {
		return 0
	}
	return len(t.Vocab)
}

var (
	_byteEncoder     map[byte]rune
	_byteEncoderOnce sync.Once
)

func getByteEncoder() map[byte]rune {
	_byteEncoderOnce.Do(func() {
		_byteEncoder = make(map[byte]rune)
		// Standard visible ASCII + Latin-1 supplement
		n := 0
		bs := make([]int, 0, 256)
		for i := int('!'); i <= int('~'); i++ {
			bs = append(bs, i)
		}
		for i := int('¡'); i <= int('¬'); i++ {
			bs = append(bs, i)
		}
		for i := int('®'); i <= int('ÿ'); i++ {
			bs = append(bs, i)
		}
		sort.Ints(bs)
		bsSet := map[int]bool{}
		for _, b := range bs {
			bsSet[b] = true
			_byteEncoder[byte(b)] = rune(b)
		}
		n = 256
		for i := 0; i < 256; i++ {
			if !bsSet[i] {
				_byteEncoder[byte(i)] = rune(n)
				n++
			}
		}
	})
	return _byteEncoder
}

var (
	_byteDecoder     map[rune]byte
	_byteDecoderOnce sync.Once
)

func getByteDecoder() map[rune]byte {
	_byteDecoderOnce.Do(func() {
		enc := getByteEncoder()
		_byteDecoder = make(map[rune]byte, len(enc))
		for b, r := range enc {
			_byteDecoder[r] = b
		}
	})
	return _byteDecoder
}
