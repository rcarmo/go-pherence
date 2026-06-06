package ideogram4

import (
	"fmt"
	"path/filepath"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

const DefaultMaxTextTokens = 2048

// PromptTokens is the tokenizer/chat-template side of Ideogram4 conditioning.
type PromptTokens struct {
	IDs       []int
	MaxTokens int
}

// LoadTokenizer loads the Qwen2-compatible tokenizer from an Ideogram4
// Diffusers directory.
func LoadTokenizer(modelDir string) (*tokenizer.Tokenizer, error) {
	if modelDir == "" {
		return nil, fmt.Errorf("empty Ideogram4 model directory")
	}
	return tokenizer.Load(filepath.Join(modelDir, "tokenizer", "tokenizer.json"))
}

// RenderChatPrompt wraps a raw prompt in the Qwen3-VL ChatML template used by
// the reference pipeline (_tokenize): a single user text message with a
// generation prompt, no system message and no tools. This matches
// apply_chat_template(messages, add_generation_prompt=True) for
// messages=[{role:user, content:[{type:text, text:prompt}]}].
func RenderChatPrompt(prompt string) string {
	return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
}

// TokenizeChatPrompt builds Qwen3-VL ChatML token ids for a single user prompt,
// assigning the control markers their exact added-token ids (the lightweight
// BPE encoder does not segment glued special tokens) and BPE-encoding only the
// textual segments.
func TokenizeChatPrompt(tok *tokenizer.Tokenizer, prompt string, maxTokens int) (PromptTokens, error) {
	if tok == nil {
		return PromptTokens{}, fmt.Errorf("nil Ideogram4 tokenizer")
	}
	if maxTokens <= 0 {
		maxTokens = DefaultMaxTextTokens
	}
	segments := []struct {
		text    string
		special bool
	}{
		{"<|im_start|>", true},
		{"user\n", false},
		{prompt, false},
		{"<|im_end|>", true},
		{"\n", false},
		{"<|im_start|>", true},
		{"assistant\n", false},
	}
	var ids []int
	for _, s := range segments {
		if s.special {
			id, ok := tok.Vocab[s.text]
			if !ok {
				return PromptTokens{}, fmt.Errorf("Ideogram4 tokenizer missing special token %q", s.text)
			}
			ids = append(ids, id)
			continue
		}
		ids = append(ids, tok.Encode(s.text)...)
	}
	if len(ids) > maxTokens {
		ids = ids[:maxTokens]
	}
	return PromptTokens{IDs: ids, MaxTokens: maxTokens}, nil
}

// TokenizePrompt tokenizes a prompt with the existing HF tokenizer loader. Full
// Qwen chat-template rendering is a later runtime step; this helper deliberately
// owns only bounded token IDs and shape validation.
func TokenizePrompt(tok *tokenizer.Tokenizer, prompt string, maxTokens int) (PromptTokens, error) {
	if tok == nil {
		return PromptTokens{}, fmt.Errorf("nil Ideogram4 tokenizer")
	}
	if maxTokens <= 0 {
		maxTokens = DefaultMaxTextTokens
	}
	ids := tok.Encode(prompt)
	if len(ids) > maxTokens {
		ids = ids[:maxTokens]
	}
	return PromptTokens{IDs: ids, MaxTokens: maxTokens}, nil
}

// ValidateTextConditioning checks the Qwen3-VL selected-hidden-state feature
// tensor shape expected by Ideogram4.
func (c Config) ValidateTextConditioning(cond TextConditioning) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if cond.Tokens < 0 {
		return fmt.Errorf("invalid Ideogram4 conditioning token count %d", cond.Tokens)
	}
	if cond.Dim != c.LLMFeaturesDim {
		return fmt.Errorf("invalid Ideogram4 conditioning dim=%d want=%d", cond.Dim, c.LLMFeaturesDim)
	}
	want := cond.Tokens * cond.Dim
	if want < 0 || len(cond.Features) != want {
		return fmt.Errorf("invalid Ideogram4 conditioning features len=%d want=%d", len(cond.Features), want)
	}
	return nil
}

// NewEmptyTextConditioning allocates a correctly-shaped placeholder feature
// tensor. It is useful for wiring scheduler/DiT shape paths before Qwen3-VL
// forward execution is implemented.
func (c Config) NewEmptyTextConditioning(tokens int) (TextConditioning, error) {
	if tokens < 0 {
		return TextConditioning{}, fmt.Errorf("invalid Ideogram4 token count %d", tokens)
	}
	if err := c.Validate(); err != nil {
		return TextConditioning{}, err
	}
	return TextConditioning{Tokens: tokens, Dim: c.LLMFeaturesDim, Features: make([]float32, tokens*c.LLMFeaturesDim)}, nil
}
