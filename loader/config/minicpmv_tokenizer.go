package config

import (
	"encoding/json"
	"os"
	"path/filepath"
)

type MiniCPMVTokenizerConfig struct {
	TokenizerClass string `json:"tokenizer_class"`
	ChatTemplate   string `json:"chat_template,omitempty"`
	BOSToken       any    `json:"bos_token"`
	EOSToken       any    `json:"eos_token"`
	PadToken       any    `json:"pad_token"`
	UNKToken       any    `json:"unk_token"`
	ImageToken     string `json:"image_token"`
	ImageStart     string `json:"im_start"`
	ImageEnd       string `json:"im_end"`
	ImagePatch     string `json:"im_patch"`
}

type MiniCPMVTokenizerMetadata struct {
	TokenizerClass    string         `json:"tokenizer_class,omitempty"`
	BOS               string         `json:"bos,omitempty"`
	EOS               string         `json:"eos,omitempty"`
	Pad               string         `json:"pad,omitempty"`
	UNK               string         `json:"unk,omitempty"`
	Image             string         `json:"image,omitempty"`
	ImageStart        string         `json:"image_start,omitempty"`
	ImageEnd          string         `json:"image_end,omitempty"`
	ImagePatch        string         `json:"image_patch,omitempty"`
	TokenIDs          map[string]int `json:"token_ids,omitempty"`
	ChatTemplateBytes int            `json:"chat_template_bytes,omitempty"`
}

func ReadMiniCPMVTokenizerMetadata(dir string) (MiniCPMVTokenizerMetadata, bool, error) {
	var out MiniCPMVTokenizerMetadata
	seen := false
	var tok MiniCPMVTokenizerConfig
	if ok, err := ReadOptionalJSON(filepath.Join(dir, "tokenizer_config.json"), &tok); err != nil {
		return out, false, err
	} else if ok {
		seen = true
		out.TokenizerClass = tok.TokenizerClass
		out.BOS = tokenString(tok.BOSToken)
		out.EOS = tokenString(tok.EOSToken)
		out.Pad = tokenString(tok.PadToken)
		out.UNK = tokenString(tok.UNKToken)
		out.Image = tok.ImageToken
		out.ImageStart = tok.ImageStart
		out.ImageEnd = tok.ImageEnd
		out.ImagePatch = tok.ImagePatch
		out.ChatTemplateBytes = len(tok.ChatTemplate)
	}
	ids, ok, err := readMiniCPMVTokenIDs(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return out, false, err
	}
	if ok {
		seen = true
		out.TokenIDs = ids
	}
	return out, seen, nil
}

func tokenString(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case map[string]any:
		if s, ok := x["content"].(string); ok {
			return s
		}
	}
	return ""
}

func readMiniCPMVTokenIDs(path string) (map[string]int, bool, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, err
	}
	var raw struct {
		AddedTokens []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
		Model struct {
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, false, err
	}
	want := map[string]bool{"<image>": true, "<im_start>": true, "<im_end>": true, "<im_patch>": true, "<|im_start|>": true, "<|im_end|>": true}
	ids := map[string]int{}
	for _, t := range raw.AddedTokens {
		if want[t.Content] {
			ids[t.Content] = t.ID
		}
	}
	for tok, id := range raw.Model.Vocab {
		if want[tok] {
			ids[tok] = id
		}
	}
	return ids, true, nil
}
