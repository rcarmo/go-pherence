package pockettts

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
	"unicode/utf8"
)

type unigramNode struct {
	next  map[byte]*unigramNode
	id    int
	score float64
}

type pocketSpecialToken struct {
	text string
	id   int
}

// Tokenizer implements the released Pocket TTS Unigram/Metaspace tokenizer,
// including UTF-8 byte fallback. It emits no BOS/EOS tokens.
type Tokenizer struct {
	root      *unigramNode
	specials  []pocketSpecialToken
	byteIDs   [256]int
	unkID     int
	unkScore  float64
	vocabSize int
}

func LoadTokenizer(r io.Reader) (*Tokenizer, error) {
	var raw struct {
		Normalizer struct {
			Type    string `json:"type"`
			Prepend string `json:"prepend"`
		} `json:"normalizer"`
		PreTokenizer struct {
			Type          string `json:"type"`
			Replacement   string `json:"replacement"`
			PrependScheme string `json:"prepend_scheme"`
			Split         bool   `json:"split"`
		} `json:"pre_tokenizer"`
		PostProcessor json.RawMessage `json:"post_processor"`
		Model         struct {
			Type         string            `json:"type"`
			UnkID        int               `json:"unk_id"`
			ByteFallback bool              `json:"byte_fallback"`
			Vocab        []json.RawMessage `json:"vocab"`
		} `json:"model"`
		Added []struct {
			ID         int    `json:"id"`
			Content    string `json:"content"`
			Special    bool   `json:"special"`
			Normalized bool   `json:"normalized"`
			LStrip     bool   `json:"lstrip"`
			RStrip     bool   `json:"rstrip"`
			SingleWord bool   `json:"single_word"`
		} `json:"added_tokens"`
	}
	dec := json.NewDecoder(r)
	if err := dec.Decode(&raw); err != nil {
		return nil, fmt.Errorf("parse Pocket TTS tokenizer: %w", err)
	}
	var trailing any
	if err := dec.Decode(&trailing); err != io.EOF {
		return nil, fmt.Errorf("Pocket TTS tokenizer contains trailing data")
	}
	if raw.Normalizer.Type != "Prepend" || raw.Normalizer.Prepend != "▁" || raw.PreTokenizer.Type != "Metaspace" || raw.PreTokenizer.Replacement != "▁" || raw.PreTokenizer.PrependScheme != "always" || !raw.PreTokenizer.Split || string(raw.PostProcessor) != "null" {
		return nil, fmt.Errorf("unsupported Pocket TTS tokenizer normalization contract")
	}
	if raw.Model.Type != "Unigram" || !raw.Model.ByteFallback || raw.Model.UnkID < 0 || raw.Model.UnkID >= len(raw.Model.Vocab) {
		return nil, fmt.Errorf("unsupported Pocket TTS tokenizer model")
	}
	t := &Tokenizer{root: &unigramNode{id: -1}, unkID: raw.Model.UnkID, vocabSize: len(raw.Model.Vocab)}
	for i := range t.byteIDs {
		t.byteIDs[i] = -1
	}
	minScore := math.Inf(1)
	for id, entry := range raw.Model.Vocab {
		var pair []json.RawMessage
		if err := json.Unmarshal(entry, &pair); err != nil || len(pair) != 2 {
			return nil, fmt.Errorf("invalid Pocket TTS vocabulary entry %d", id)
		}
		var piece string
		var score float64
		if err := json.Unmarshal(pair[0], &piece); err != nil {
			return nil, err
		}
		if err := json.Unmarshal(pair[1], &score); err != nil {
			return nil, err
		}
		if piece == "" || math.IsNaN(score) || math.IsInf(score, 0) {
			return nil, fmt.Errorf("invalid Pocket TTS vocabulary value %d", id)
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
			return nil, fmt.Errorf("duplicate Pocket TTS vocabulary piece %q", piece)
		}
		node.id, node.score = id, score
		var value int
		if _, err := fmt.Sscanf(piece, "<0x%02X>", &value); err == nil && value >= 0 && value < 256 && piece == fmt.Sprintf("<0x%02X>", value) {
			t.byteIDs[value] = id
		}
	}
	t.unkScore = minScore - 10
	for value, id := range t.byteIDs {
		if id < 0 {
			return nil, fmt.Errorf("Pocket TTS tokenizer missing byte fallback %02X", value)
		}
	}
	seenID, seenText := map[int]bool{}, map[string]bool{}
	for _, added := range raw.Added {
		if added.ID < 0 || added.ID >= t.vocabSize || added.Content == "" || !added.Special || added.Normalized || added.LStrip || added.RStrip || added.SingleWord || seenID[added.ID] || seenText[added.Content] {
			return nil, fmt.Errorf("unsupported Pocket TTS added token %q", added.Content)
		}
		seenID[added.ID], seenText[added.Content] = true, true
		t.specials = append(t.specials, pocketSpecialToken{text: added.Content, id: added.ID})
	}
	sort.SliceStable(t.specials, func(i, j int) bool { return len(t.specials[i].text) > len(t.specials[j].text) })
	return t, nil
}

func (t *Tokenizer) VocabSize() int {
	if t == nil {
		return 0
	}
	return t.vocabSize
}

func (t *Tokenizer) Encode(text string) ([]uint32, error) {
	if t == nil || t.root == nil {
		return nil, fmt.Errorf("nil Pocket TTS tokenizer")
	}
	if !utf8.ValidString(text) {
		return nil, fmt.Errorf("Pocket TTS text is not UTF-8")
	}
	if text == "" {
		return []uint32{}, nil
	}
	var ids []uint32
	for len(text) > 0 {
		at := len(text)
		var chosen *pocketSpecialToken
		for i := range t.specials {
			if index := strings.Index(text, t.specials[i].text); index >= 0 && index < at {
				at, chosen = index, &t.specials[i]
			}
		}
		if at > 0 {
			ids = append(ids, t.encodeOrdinary(text[:at])...)
		}
		if chosen == nil {
			break
		}
		ids = append(ids, uint32(chosen.id))
		text = text[at+len(chosen.text):]
	}
	return ids, nil
}

func (t *Tokenizer) encodeOrdinary(text string) []uint32 {
	// The released JSON applies Prepend("▁") and then Metaspace replacement.
	transformed := "▁" + strings.ReplaceAll(text, " ", "▁")
	n := len(transformed)
	best := make([]float64, n+1)
	prev := make([]int, n+1)
	token := make([]int, n+1)
	unknown := make([]bool, n+1)
	for i := 1; i <= n; i++ {
		best[i] = math.Inf(-1)
		prev[i] = -1
	}
	for i := 0; i < n; {
		if math.IsInf(best[i], -1) {
			_, size := utf8.DecodeRuneInString(transformed[i:])
			i += size
			continue
		}
		_, size := utf8.DecodeRuneInString(transformed[i:])
		node := t.root
		matchedRune := false
		for j := i; j < n; j++ {
			node = node.next[transformed[j]]
			if node == nil {
				break
			}
			if node.id >= 0 {
				end := j + 1
				if end-i == size {
					matchedRune = true
				}
				score := best[i] + node.score
				if score > best[end] {
					best[end], prev[end], token[end], unknown[end] = score, i, node.id, false
				}
			}
		}
		if !matchedRune {
			end := i + size
			score := best[i] + t.unkScore
			if score > best[end] {
				best[end], prev[end], token[end], unknown[end] = score, i, t.unkID, true
			}
		}
		i += size
	}
	if prev[n] < 0 {
		return []uint32{uint32(t.unkID)}
	}
	type piece struct {
		start, end, id int
		unknown        bool
	}
	var reverse []piece
	for pos := n; pos > 0; pos = prev[pos] {
		reverse = append(reverse, piece{prev[pos], pos, token[pos], unknown[pos]})
	}
	ids := make([]uint32, 0, len(reverse))
	for i := len(reverse) - 1; i >= 0; i-- {
		p := reverse[i]
		if p.unknown {
			for _, value := range []byte(transformed[p.start:p.end]) {
				ids = append(ids, uint32(t.byteIDs[value]))
			}
		} else {
			ids = append(ids, uint32(p.id))
		}
	}
	return ids
}
