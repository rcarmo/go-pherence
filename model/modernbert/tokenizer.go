package modernbert

import (
	"fmt"
	tokloader "github.com/rcarmo/go-pherence/loader/tokenizer"
	"path/filepath"
)

type Tokenizer struct {
	inner               *tokloader.Tokenizer
	CLS, SEP, Pad, Mask int
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
	return &Tokenizer{inner: t, CLS: cls, SEP: sep, Pad: pad, Mask: mask}, nil
}
func (t *Tokenizer) Encode(text string, addSpecial bool) ([]int, error) {
	if t == nil || t.inner == nil {
		return nil, fmt.Errorf("modernbert: nil tokenizer")
	}
	ids := t.inner.Encode(text)
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
