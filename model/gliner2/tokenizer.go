package gliner2

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

type unigramNode struct {
	next  map[byte]*unigramNode
	id    int
	score float64
}
type specialToken struct {
	text string
	id   int
}

// Tokenizer implements the published GLiNER2.5 Unigram/Metaspace contract.
// Input offsets are not returned: schema word alignment is a separate step.
type Tokenizer struct {
	root     *unigramNode
	specials []specialToken
	unk      int
	unkScore float64
	cls, sep int
}

type metaspaceConfig struct {
	Type, Replacement string
	Prepend           string `json:"prepend_scheme"`
	Split             bool
	Pretokenizers     []metaspaceConfig `json:"pretokenizers"`
}

func LoadTokenizer(r io.Reader) (*Tokenizer, error) {
	var raw struct {
		Model struct {
			Type         string            `json:"type"`
			UnkID        int               `json:"unk_id"`
			ByteFallback bool              `json:"byte_fallback"`
			Vocab        []json.RawMessage `json:"vocab"`
		} `json:"model"`
		Normalizer json.RawMessage `json:"normalizer"`
		Pre        metaspaceConfig `json:"pre_tokenizer"`
		Added      []struct {
			ID                  int
			Content             string
			Special, Normalized bool
			LStrip              bool `json:"lstrip"`
			RStrip              bool `json:"rstrip"`
			SingleWord          bool `json:"single_word"`
		} `json:"added_tokens"`
		Post json.RawMessage `json:"post_processor"`
	}
	d := json.NewDecoder(r)
	if err := d.Decode(&raw); err != nil {
		return nil, err
	}
	if raw.Pre.Type == "Sequence" && len(raw.Pre.Pretokenizers) == 1 {
		raw.Pre = raw.Pre.Pretokenizers[0]
	}
	if raw.Model.Type != "Unigram" || raw.Model.ByteFallback || raw.Pre.Type != "Metaspace" || raw.Pre.Replacement != "▁" || raw.Pre.Prepend != "always" || !raw.Pre.Split {
		return nil, fmt.Errorf("unsupported tokenizer model/pretokenizer")
	}
	var normal struct {
		Type        string
		Normalizers []struct {
			Type       string
			Pattern    map[string]string
			Content    string
			StripLeft  bool `json:"strip_left"`
			StripRight bool `json:"strip_right"`
		}
	}
	if err := json.Unmarshal(raw.Normalizer, &normal); err != nil {
		return nil, err
	}
	if normal.Type != "Sequence" || len(normal.Normalizers) != 3 {
		return nil, fmt.Errorf("unsupported normalizer")
	}
	ns := normal.Normalizers
	if ns[0].Type != "Replace" || ns[0].Pattern["Regex"] != `\s{2,}|[\n\r\t]` || ns[0].Content != " " || ns[1].Type != "NFC" || ns[2].Type != "Strip" || ns[2].StripLeft || !ns[2].StripRight {
		return nil, fmt.Errorf("unsupported normalization sequence")
	}
	if raw.Model.UnkID < 0 || raw.Model.UnkID >= len(raw.Model.Vocab) {
		return nil, fmt.Errorf("invalid unknown ID")
	}
	t := &Tokenizer{root: &unigramNode{id: -1}, unk: raw.Model.UnkID, cls: -1, sep: -1}
	minScore := math.Inf(1)
	for id, entry := range raw.Model.Vocab {
		var pair []json.RawMessage
		if err := json.Unmarshal(entry, &pair); err != nil || len(pair) != 2 {
			return nil, fmt.Errorf("invalid vocabulary entry %d", id)
		}
		var piece string
		var score float64
		if err := json.Unmarshal(pair[0], &piece); err != nil {
			return nil, err
		}
		if err := json.Unmarshal(pair[1], &score); err != nil {
			return nil, err
		}
		if piece == "" || math.IsInf(score, 0) || math.IsNaN(score) {
			return nil, fmt.Errorf("invalid vocabulary value")
		}
		minScore = math.Min(minScore, score)
		node := t.root
		for i := 0; i < len(piece); i++ {
			if node.next == nil {
				node.next = make(map[byte]*unigramNode)
			}
			if node.next[piece[i]] == nil {
				node.next[piece[i]] = &unigramNode{id: -1}
			}
			node = node.next[piece[i]]
		}
		if node.id >= 0 {
			return nil, fmt.Errorf("duplicate vocabulary piece")
		}
		node.id = id
		node.score = score
	}
	t.unkScore = minScore - 10
	for _, a := range raw.Added {
		if !a.Special || a.Normalized || a.LStrip || a.RStrip || a.SingleWord || a.Content == "" || a.ID < 0 {
			return nil, fmt.Errorf("unsupported added token %q", a.Content)
		}
		t.specials = append(t.specials, specialToken{a.Content, a.ID})
		if a.Content == "[CLS]" {
			t.cls = a.ID
		}
		if a.Content == "[SEP]" {
			t.sep = a.ID
		}
	}
	sort.SliceStable(t.specials, func(i, j int) bool { return len(t.specials[i].text) > len(t.specials[j].text) })
	var post struct {
		Type   string
		Single []json.RawMessage
	}
	if err := json.Unmarshal(raw.Post, &post); err != nil {
		return nil, err
	}
	if post.Type != "TemplateProcessing" || len(post.Single) != 3 || t.cls < 0 || t.sep < 0 {
		return nil, fmt.Errorf("unsupported single sequence template")
	}
	// Validate actual template rather than accepting arbitrary templates by type.
	var first, last struct {
		Special struct {
			ID string `json:"id"`
		} `json:"SpecialToken"`
	}
	var middle struct {
		Sequence struct {
			ID string `json:"id"`
		} `json:"Sequence"`
	}
	json.Unmarshal(post.Single[0], &first)
	json.Unmarshal(post.Single[1], &middle)
	json.Unmarshal(post.Single[2], &last)
	if first.Special.ID != "[CLS]" || middle.Sequence.ID != "A" || last.Special.ID != "[SEP]" {
		return nil, fmt.Errorf("unsupported template ordering")
	}
	return t, nil
}

func (t *Tokenizer) Encode(text string, addSpecial bool) ([]int, error) {
	if t == nil || t.root == nil {
		return nil, fmt.Errorf("nil tokenizer")
	}
	if !utf8.ValidString(text) {
		return nil, fmt.Errorf("invalid UTF-8")
	}
	ids := []int{}
	if addSpecial {
		ids = append(ids, t.cls)
	}
	// Split literal specials before normalisation, matching AddedVocabulary.
	for len(text) > 0 {
		at := len(text)
		var chosen *specialToken
		for i := range t.specials {
			p := strings.Index(text, t.specials[i].text)
			if p >= 0 && p < at {
				at = p
				chosen = &t.specials[i]
			}
		}
		if at > 0 {
			ids = append(ids, t.encodeText(text[:at])...)
		}
		if chosen == nil {
			break
		}
		ids = append(ids, chosen.id)
		text = text[at+len(chosen.text):]
	}
	if addSpecial {
		ids = append(ids, t.sep)
	}
	return ids, nil
}
func (t *Tokenizer) encodeText(text string) []int {
	// Go regexp's \s is ASCII; expand Unicode whitespace runs explicitly.
	runes := []rune(text)
	var b strings.Builder
	for i := 0; i < len(runes); {
		if unicode.IsSpace(runes[i]) {
			j := i + 1
			for j < len(runes) && unicode.IsSpace(runes[j]) {
				j++
			}
			if j-i >= 2 || runes[i] == '\n' || runes[i] == '\r' || runes[i] == '\t' {
				b.WriteByte(' ')
			} else {
				b.WriteRune(runes[i])
			}
			i = j
		} else {
			b.WriteRune(runes[i])
			i++
		}
	}
	text = strings.TrimRightFunc(norm.NFC.String(b.String()), unicode.IsSpace)
	if text == "" {
		return nil
	}
	text = strings.ReplaceAll(text, " ", "▁")
	if !strings.HasPrefix(text, "▁") {
		text = "▁" + text
	}
	// Metaspace split merges each delimiter with the following word.
	parts := strings.Split(text, "▁")
	var ids []int
	for _, p := range parts[1:] {
		ids = append(ids, t.segment("▁"+p)...)
	}
	return ids
}
func (t *Tokenizer) segment(s string) []int {
	n := len(s)
	best := make([]float64, n+1)
	prev := make([]int, n+1)
	token := make([]int, n+1)
	for i := 1; i <= n; i++ {
		best[i] = math.Inf(-1)
	}
	for i := 0; i < n; {
		_, size := utf8.DecodeRuneInString(s[i:])
		node := t.root
		hasSingle := false
		for j := i; j < n; j++ {
			node = node.next[s[j]]
			if node == nil {
				break
			}
			if node.id >= 0 {
				end := j + 1
				if end-i == size {
					hasSingle = true
				}
				v := best[i] + node.score
				if v > best[end] {
					best[end] = v
					prev[end] = i
					token[end] = node.id
				}
			}
		}
		if !hasSingle {
			end := i + size
			v := best[i] + t.unkScore
			if v > best[end] {
				best[end] = v
				prev[end] = i
				token[end] = t.unk
			}
		}
		i += size
	}
	var reverse []int
	for pos := n; pos > 0; pos = prev[pos] {
		reverse = append(reverse, token[pos])
	}
	ids := make([]int, 0, len(reverse))
	for i := len(reverse) - 1; i >= 0; i-- {
		id := reverse[i]
		if id == t.unk && len(ids) > 0 && ids[len(ids)-1] == id {
			continue
		}
		ids = append(ids, id)
	}
	return ids
}
