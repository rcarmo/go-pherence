package tokenizer

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"
	"sync"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// Tokenizer handles BPE tokenization for LLaMA-style models.
type Tokenizer struct {
	Vocab        map[string]int // token string → ID
	InvVocab     map[int]string // ID → token string
	Merges       [][2]string    // BPE merge pairs in priority order
	AddedSpecial map[string]int // Hugging Face added tokens with special=true
	AddedTokens  map[string]int // exact non-special added tokens from sidecars

	mergeRankOnce sync.Once
	mergeRank     map[[2]string]int
	normalizer    tokenizerNormalizer
	byteLevelMode byteLevelPretokenizer
}

type tokenizerNormalizer uint8

const (
	normalizerNone tokenizerNormalizer = iota
	normalizerNFC
)

type byteLevelPretokenizer uint8

const (
	byteLevelDefault byteLevelPretokenizer = iota
	byteLevelQwenSingleDigits
)

const (
	qwenDigitsRunRegex    = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+`
	qwenSingleDigitSource = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	qwenSingleDigitRegex  = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+`
)

var (
	qwenDigitsRunPattern   = regexp.MustCompile(qwenDigitsRunRegex)
	qwenSingleDigitPattern = regexp.MustCompile(qwenSingleDigitRegex)
)

// Load loads a HuggingFace tokenizer.json.
func Load(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var raw struct {
		PreTokenizer struct {
			Type          string `json:"type"`
			Pretokenizers []struct {
				Type    string `json:"type"`
				Pattern struct {
					Regex string `json:"Regex"`
				} `json:"pattern"`
			} `json:"pretokenizers"`
		} `json:"pre_tokenizer"`
		Normalizer json.RawMessage `json:"normalizer"`
		Model      struct {
			Vocab  map[string]int  `json:"vocab"`
			Merges json.RawMessage `json:"merges"`
		} `json:"model"`
		AddedTokens []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
			Special bool   `json:"special"`
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}

	if raw.Model.Vocab == nil {
		raw.Model.Vocab = map[string]int{}
	}
	t := &Tokenizer{
		Vocab:         raw.Model.Vocab,
		InvVocab:      make(map[int]string, len(raw.Model.Vocab)),
		AddedSpecial:  make(map[string]int),
		normalizer:    detectNormalizer(raw.Normalizer),
		byteLevelMode: detectByteLevelPretokenizer(raw.PreTokenizer.Pretokenizers),
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
		if at.Special {
			t.AddedSpecial[at.Content] = at.ID
		}
	}

	if len(raw.Model.Merges) == 0 || string(raw.Model.Merges) == "null" {
		return t, nil
	}

	// Select the representation before decoding: attempting []string first
	// allocates an error for every array entry in large Qwen tokenizers.
	merges := bytes.TrimSpace(raw.Model.Merges)
	if len(merges) < 2 || merges[0] != '[' {
		return nil, fmt.Errorf("unsupported merges format")
	}
	first := bytes.TrimSpace(merges[1:])
	count := jsonArrayLen(merges)
	if len(first) > 0 && first[0] == '"' {
		mergeStrings := make([]string, 0, count)
		if err := json.Unmarshal(merges, &mergeStrings); err != nil {
			return nil, fmt.Errorf("unsupported merges format: %w", err)
		}
		t.Merges = make([][2]string, len(mergeStrings))
		for i, m := range mergeStrings {
			a, b, ok := strings.Cut(m, " ")
			if !ok || a == "" || b == "" {
				return nil, fmt.Errorf("malformed merge at index %d", i)
			}
			t.Merges[i] = [2]string{a, b}
		}
	} else {
		t.Merges = make([][2]string, 0, count)
		if err := json.Unmarshal(merges, &t.Merges); err != nil {
			return nil, fmt.Errorf("unsupported merges format: %w", err)
		}
		for i, m := range t.Merges {
			if m[0] == "" || m[1] == "" {
				return nil, fmt.Errorf("malformed merge at index %d", i)
			}
		}
	}

	t.initMergeRank()
	return t, nil
}

// jsonArrayLen counts top-level entries in an already validated JSON array.
// Load's outer json.Unmarshal validates syntax before this allocation-sizing
// pass. Strings (including escapes) and nested arrays/objects do not add entries.
func jsonArrayLen(data []byte) int {
	depth, count := 0, 0
	quoted := false
	for i := 0; i < len(data); i++ {
		c := data[i]
		if quoted {
			if c == '\\' {
				i++
			} else if c == '"' {
				quoted = false
			}
			continue
		}
		switch c {
		case ' ', '\t', '\r', '\n':
			continue
		case ',':
			if depth == 1 {
				count++
			}
		case ']', '}':
			depth--
		default:
			if depth == 1 && count == 0 {
				count = 1
			}
			if c == '"' {
				quoted = true
			} else if c == '[' || c == '{' {
				depth++
			}
		}
	}
	return count
}

func detectNormalizer(raw json.RawMessage) tokenizerNormalizer {
	if len(raw) == 0 || string(raw) == "null" {
		return normalizerNone
	}
	var probe struct {
		Type        string            `json:"type"`
		Normalizers []json.RawMessage `json:"normalizers"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		return normalizerNone
	}
	switch probe.Type {
	case "NFC":
		return normalizerNFC
	case "Sequence":
		if len(probe.Normalizers) == 0 {
			return normalizerNone
		}
		for _, part := range probe.Normalizers {
			if detectNormalizer(part) != normalizerNFC {
				return normalizerNone
			}
		}
		return normalizerNFC
	default:
		return normalizerNone
	}
}

func detectByteLevelPretokenizer(parts []struct {
	Type    string `json:"type"`
	Pattern struct {
		Regex string `json:"Regex"`
	} `json:"pattern"`
}) byteLevelPretokenizer {
	for _, part := range parts {
		if part.Type != "Split" {
			continue
		}
		switch part.Pattern.Regex {
		case qwenSingleDigitSource, qwenSingleDigitRegex:
			return byteLevelQwenSingleDigits
		}
		if strings.Contains(part.Pattern.Regex, `|\p{N}|`) {
			return byteLevelQwenSingleDigits
		}
	}
	return byteLevelDefault
}

func (t *Tokenizer) normalizeOrdinary(text string) string {
	if t == nil || text == "" {
		return text
	}
	switch t.normalizer {
	case normalizerNFC:
		if norm.NFC.IsNormalString(text) {
			return text
		}
		return norm.NFC.String(text)
	default:
		return text
	}
}

func (t *Tokenizer) initMergeRank() {
	if t == nil {
		return
	}
	t.mergeRankOnce.Do(func() {
		if len(t.Merges) == 0 {
			return
		}
		t.mergeRank = make(map[[2]string]int, len(t.Merges))
		for i, m := range t.Merges {
			t.mergeRank[m] = i
		}
	})
}

// Encode tokenizes a string into token IDs.
func (t *Tokenizer) Encode(text string) []int {
	if t == nil || t.Vocab == nil {
		return nil
	}
	if len(t.AddedSpecial) == 0 && len(t.AddedTokens) == 0 {
		return t.encodeOrdinary(text)
	}
	var ids []int
	for len(text) > 0 {
		next := len(text)
		nextToken := ""
		nextID := 0
		for _, tokens := range []map[string]int{t.AddedSpecial, t.AddedTokens} {
			for token, id := range tokens {
				if token == "" {
					continue
				}
				if at := strings.Index(text, token); at >= 0 && (at < next || at == next && len(token) > len(nextToken)) {
					next, nextToken, nextID = at, token, id
				}
			}
		}
		if nextToken == "" {
			ids = append(ids, t.encodeOrdinary(text)...)
			break
		}
		ids = append(ids, t.encodeOrdinary(text[:next])...)
		ids = append(ids, nextID)
		text = text[next+len(nextToken):]
	}
	return ids
}

// ValidateUserText rejects caller-controlled text that would be recognized as
// a configured special token. Trusted prompt renderers may then insert those
// control tokens separately with Encode.
func (t *Tokenizer) ValidateUserText(text string) error {
	if t == nil {
		return fmt.Errorf("nil tokenizer")
	}
	for token := range t.AddedSpecial {
		if token != "" && strings.Contains(text, token) {
			return fmt.Errorf("text contains reserved tokenizer token")
		}
	}
	return nil
}

func (t *Tokenizer) encodeOrdinary(text string) []int {
	if text == "" {
		return nil
	}
	text = t.normalizeOrdinary(text)
	if text == "" {
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

func (t *Tokenizer) byteLevelPattern() *regexp.Regexp {
	if t != nil && t.byteLevelMode == byteLevelQwenSingleDigits {
		return qwenSingleDigitPattern
	}
	return qwenDigitsRunPattern
}

// encodeByteLevel performs faithful Qwen byte-level BPE: pre-tokenization,
// per-byte unicode mapping, then rank-ordered merges over the byte symbols
// (matching the inverse applied by Decode).
func (t *Tokenizer) encodeByteLevel(text string) []int {
	t.initMergeRank()
	mergeRank := t.mergeRank
	byteEncoder := getByteEncoder()

	pieces := splitWhitespaceRuns(t.byteLevelPattern().FindAllString(text, -1))
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

// encodeSentencePiece implements the Gemma tokenizer.json contract: replace
// literal spaces with the SentencePiece marker, then run BPE over the intact
// rune stream. Splitting with strings.Fields is incorrect because it discards
// leading/repeated spaces, newlines and tabs before tokenization.
func (t *Tokenizer) encodeSentencePiece(text string) []int {
	text = strings.ReplaceAll(text, " ", "\u2581")
	if text == "" {
		return nil
	}
	if id, ok := t.Vocab[text]; ok {
		return []int{id}
	}
	t.initMergeRank()
	chars := make([]string, 0, len(text))
	for _, r := range text {
		chars = append(chars, string(r))
	}
	for len(chars) >= 2 {
		bestRank := len(t.Merges)
		bestIdx := -1
		for i := 0; i < len(chars)-1; i++ {
			if rank, ok := t.mergeRank[[2]string{chars[i], chars[i+1]}]; ok && rank < bestRank {
				bestRank = rank
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break
		}
		chars[bestIdx] += chars[bestIdx+1]
		copy(chars[bestIdx+1:], chars[bestIdx+2:])
		chars = chars[:len(chars)-1]
	}
	ids := make([]int, 0, len(chars))
	for _, symbol := range chars {
		if id, ok := t.Vocab[symbol]; ok {
			ids = append(ids, id)
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
