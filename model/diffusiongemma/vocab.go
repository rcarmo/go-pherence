package diffusiongemma

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

type Vocab struct {
	TokenToID map[string]int `json:"-"`
	IDToToken map[int]string `json:"-"`
}

func LoadVocab(modelDir string) (*Vocab, error) {
	path := filepath.Join(modelDir, "tokenizer.json")
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw hfTokenizerJSON
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, err
	}
	v := &Vocab{TokenToID: map[string]int{}, IDToToken: map[int]string{}}
	for tok, id := range raw.Model.Vocab {
		v.TokenToID[tok] = id
		if _, exists := v.IDToToken[id]; !exists {
			v.IDToToken[id] = tok
		}
	}
	for _, tok := range raw.AddedTokens {
		v.TokenToID[tok.Content] = tok.ID
		v.IDToToken[tok.ID] = tok.Content
	}
	return v, nil
}

func (v *Vocab) EncodeExact(tokens []string) ([]int, error) {
	if v == nil {
		return nil, fmt.Errorf("nil DiffusionGemma vocab")
	}
	ids := make([]int, 0, len(tokens))
	for _, tok := range tokens {
		id, ok := v.TokenToID[tok]
		if !ok {
			return nil, fmt.Errorf("DiffusionGemma token %q not found in exact vocab", tok)
		}
		ids = append(ids, id)
	}
	return ids, nil
}

func (v *Vocab) DecodeIDs(ids []int) []string {
	if v == nil {
		return nil
	}
	out := make([]string, len(ids))
	for i, id := range ids {
		if tok, ok := v.IDToToken[id]; ok {
			out[i] = tok
		} else {
			out[i] = fmt.Sprintf("<id:%d>", id)
		}
	}
	return out
}
