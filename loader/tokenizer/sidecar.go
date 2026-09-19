package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
)

// LoadWithConfig adds exact-match tokens described only in tokenizer_config.json.
// It does not approximate whitespace/single-word/normalised AddedToken policies:
// those are rejected so callers do not silently render a different prompt.
func LoadWithConfig(dir string) (*Tokenizer, error) {
	t, err := Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(filepath.Join(dir, "tokenizer_config.json"))
	if err != nil {
		return nil, err
	}
	var cfg struct {
		Added map[string]struct {
			Content    string `json:"content"`
			Special    bool   `json:"special"`
			SingleWord bool   `json:"single_word"`
			LStrip     bool   `json:"lstrip"`
			RStrip     bool   `json:"rstrip"`
			Normalized bool   `json:"normalized"`
		} `json:"added_tokens_decoder"`
	}
	if err = json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	t.AddedTokens = map[string]int{}
	if t.AddedSpecial == nil {
		t.AddedSpecial = map[string]int{}
	}
	for key, token := range cfg.Added {
		id, err := strconv.Atoi(key)
		if err != nil || id < 0 || token.Content == "" {
			return nil, fmt.Errorf("invalid added token %q", key)
		}
		if token.SingleWord || token.LStrip || token.RStrip || token.Normalized {
			return nil, fmt.Errorf("unsupported added-token policy for %q", token.Content)
		}
		if old, ok := t.Vocab[token.Content]; ok && old != id {
			return nil, fmt.Errorf("added token ID conflict")
		}
		if old, ok := t.InvVocab[id]; ok && old != token.Content {
			return nil, fmt.Errorf("added token text conflict")
		}
		t.Vocab[token.Content] = id
		t.InvVocab[id] = token.Content
		if token.Special {
			t.AddedSpecial[token.Content] = id
		} else {
			t.AddedTokens[token.Content] = id
		}
	}
	return t, nil
}
