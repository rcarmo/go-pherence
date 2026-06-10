package diffusiongemma

import "fmt"

// SpecialTokenIDs gives typed access to the control-token IDs extracted from
// tokenizer.json. Missing tokens use -1.
type SpecialTokenIDs struct {
	BOS   int `json:"bos"`
	EOS   int `json:"eos"`
	PAD   int `json:"pad"`
	MASK  int `json:"mask"`
	THINK int `json:"think"`
	BOI   int `json:"boi"`
	EOI   int `json:"eoi"`
	IMAGE int `json:"image"`
	BOT   int `json:"bot"`
	EOT   int `json:"eot"`
	BOC   int `json:"boc"`
	EOC   int `json:"eoc"`
}

func (m TokenizerMetadata) TokenID(token string) (int, bool) {
	if m.TokenIDs == nil || token == "" {
		return 0, false
	}
	id, ok := m.TokenIDs[token]
	return id, ok
}

func (m TokenizerMetadata) SpecialTokenIDs(processor *ProcessorMetadata) SpecialTokenIDs {
	out := SpecialTokenIDs{BOS: -1, EOS: -1, PAD: -1, MASK: -1, THINK: -1, BOI: -1, EOI: -1, IMAGE: -1, BOT: -1, EOT: -1, BOC: -1, EOC: -1}
	lookup := func(tok string) int {
		if id, ok := m.TokenID(tok); ok {
			return id
		}
		return -1
	}
	if processor != nil {
		out.BOS = lookup(processor.BOS)
		out.EOS = lookup(processor.EOS)
		out.PAD = lookup(processor.Pad)
		out.MASK = lookup(processor.Mask)
		out.THINK = lookup(processor.Think)
		out.BOI = lookup(processor.BOI)
		out.EOI = lookup(processor.EOI)
		out.IMAGE = lookup(processor.Image)
		out.BOT = lookup(processor.BOT)
		out.EOT = lookup(processor.EOT)
		out.BOC = lookup(processor.BOC)
		out.EOC = lookup(processor.EOC)
	}
	return out
}

// PromptIDs is a token-ID level prompt scaffold. It deliberately avoids chat
// template rendering and text tokenization; callers provide already-tokenized
// content and this helper adds well-known framing tokens when available.
type PromptIDs struct {
	InputIDs []int `json:"input_ids"`
}

type PromptOptions struct {
	AddBOS              bool `json:"add_bos"`
	AddGenerationPrompt bool `json:"add_generation_prompt"`
	EnableThinking      bool `json:"enable_thinking"`
}

func BuildPromptIDs(content []int, specials SpecialTokenIDs, opts PromptOptions) (PromptIDs, error) {
	ids := make([]int, 0, len(content)+4)
	if opts.AddBOS {
		if specials.BOS < 0 {
			return PromptIDs{}, fmt.Errorf("DiffusionGemma BOS token ID unavailable")
		}
		ids = append(ids, specials.BOS)
	}
	if opts.EnableThinking {
		if specials.THINK < 0 {
			return PromptIDs{}, fmt.Errorf("DiffusionGemma think token ID unavailable")
		}
		ids = append(ids, specials.THINK)
	}
	ids = append(ids, content...)
	if opts.AddGenerationPrompt && specials.BOT >= 0 {
		ids = append(ids, specials.BOT)
	}
	return PromptIDs{InputIDs: ids}, nil
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content []int  `json:"content"`
}

type ChatPromptOptions struct {
	AddBOS              bool `json:"add_bos"`
	AddGenerationPrompt bool `json:"add_generation_prompt"`
	EnableThinking      bool `json:"enable_thinking"`
}

func BuildSimpleChatPromptIDs(messages []ChatMessage, specials SpecialTokenIDs, opts ChatPromptOptions) (PromptIDs, error) {
	ids := make([]int, 0)
	if opts.AddBOS {
		if specials.BOS < 0 {
			return PromptIDs{}, fmt.Errorf("DiffusionGemma BOS token ID unavailable")
		}
		ids = append(ids, specials.BOS)
	}
	if opts.EnableThinking {
		if specials.THINK < 0 {
			return PromptIDs{}, fmt.Errorf("DiffusionGemma think token ID unavailable")
		}
		ids = append(ids, specials.THINK)
	}
	for _, msg := range messages {
		if specials.BOT >= 0 {
			ids = append(ids, specials.BOT)
		}
		// Role strings are intentionally not tokenized here; callers should include
		// role/content token IDs once full chat-template rendering is implemented.
		ids = append(ids, msg.Content...)
		if specials.EOT >= 0 {
			ids = append(ids, specials.EOT)
		}
	}
	if opts.AddGenerationPrompt && specials.BOT >= 0 {
		ids = append(ids, specials.BOT)
	}
	return PromptIDs{InputIDs: ids}, nil
}
