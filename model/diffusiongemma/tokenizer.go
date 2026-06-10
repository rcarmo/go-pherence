package diffusiongemma

import (
	"encoding/json"
	"os"
	"path/filepath"
)

type TokenizerMetadata struct {
	Path        string         `json:"path"`
	VocabSize   int            `json:"vocab_size"`
	AddedTokens int            `json:"added_tokens"`
	TokenIDs    map[string]int `json:"token_ids,omitempty"`
}

type hfTokenizerJSON struct {
	Model struct {
		Vocab map[string]int `json:"vocab"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
	} `json:"added_tokens"`
}

func ReadTokenizerMetadata(modelDir string, tokens []string) (TokenizerMetadata, bool, error) {
	path := filepath.Join(modelDir, "tokenizer.json")
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return TokenizerMetadata{}, false, nil
		}
		return TokenizerMetadata{}, false, err
	}
	var raw hfTokenizerJSON
	if err := json.Unmarshal(b, &raw); err != nil {
		return TokenizerMetadata{}, false, err
	}
	out := TokenizerMetadata{Path: path, VocabSize: len(raw.Model.Vocab), AddedTokens: len(raw.AddedTokens), TokenIDs: map[string]int{}}
	for _, tok := range tokens {
		if id, ok := raw.Model.Vocab[tok]; ok {
			out.TokenIDs[tok] = id
		}
	}
	for _, tok := range tokens {
		if _, ok := out.TokenIDs[tok]; ok {
			continue
		}
		for _, added := range raw.AddedTokens {
			if added.Content == tok {
				out.TokenIDs[tok] = added.ID
				break
			}
		}
	}
	return out, true, nil
}
