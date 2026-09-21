package modernbert

import (
	"fmt"
	tokloader "github.com/rcarmo/go-pherence/loader/tokenizer"
	"path/filepath"
	"unicode"
)

type Tokenizer struct {
	inner               *tokloader.Tokenizer
	CLS, SEP, Pad, Mask int
	spaceID             int
	spaceNumber         map[int]int
}

func LoadTokenizer(modelDir string) (*Tokenizer, error) {
	t, e := tokloader.Load(filepath.Join(modelDir, "tokenizer.json"))
	if e != nil {
		return nil, e
	}
	find := func(text string) (int, error) {
		ids := t.Encode(text)
		if len(ids) != 1 {
			return 0, fmt.Errorf("modernbert: tokenizer special %s", text)
		}
		return ids[0], nil
	}
	cls, e := find("[CLS]")
	if e != nil {
		return nil, e
	}
	sep, e := find("[SEP]")
	if e != nil {
		return nil, e
	}
	pad, e := find("[PAD]")
	if e != nil {
		return nil, e
	}
	mask, e := find("[MASK]")
	if e != nil {
		return nil, e
	}
	spaceID, hasSpace := t.Vocab["Ġ"]
	spaceNumber := map[int]int{}
	if hasSpace {
		for token, id := range t.Vocab {
			if token == "" {
				continue
			}
			digits := true
			for _, r := range token {
				if !unicode.IsDigit(r) || r > unicode.MaxASCII {
					digits = false
					break
				}
			}
			if digits {
				if joined, ok := t.Vocab["Ġ"+token]; ok {
					spaceNumber[id] = joined
				}
			}
		}
	}
	return &Tokenizer{inner: t, CLS: cls, SEP: sep, Pad: pad, Mask: mask, spaceID: spaceID, spaceNumber: spaceNumber}, nil
}
func (t *Tokenizer) Encode(text string, addSpecial bool) ([]int, error) {
	if t == nil || t.inner == nil {
		return nil, fmt.Errorf("modernbert: nil tokenizer")
	}
	ids := t.inner.Encode(text)
	// Hugging Face's default GPT-2 ByteLevel regex includes one optional
	// leading space in number runs. The shared tokenizer's Qwen-compatible
	// default leaves that space separate, so fold the equivalent vocabulary
	// pair here without changing Qwen behavior or the frozen tokenizer source.
	if len(t.spaceNumber) > 0 {
		out := make([]int, 0, len(ids))
		for i := 0; i < len(ids); i++ {
			if ids[i] == t.spaceID && i+1 < len(ids) {
				if joined, ok := t.spaceNumber[ids[i+1]]; ok {
					out = append(out, joined)
					i++
					continue
				}
			}
			out = append(out, ids[i])
		}
		ids = out
	}
	if addSpecial {
		out := make([]int, 0, len(ids)+2)
		out = append(out, t.CLS)
		out = append(out, ids...)
		out = append(out, t.SEP)
		ids = out
	}
	return ids, nil
}
func (t *Tokenizer) Decode(ids []int) string {
	if t == nil || t.inner == nil {
		return ""
	}
	return t.inner.Decode(ids)
}
