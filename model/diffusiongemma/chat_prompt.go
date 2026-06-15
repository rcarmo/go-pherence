package diffusiongemma

import "fmt"

// EncodeTextFunc is the minimal tokenizer boundary needed by chat prompt
// scaffolding. It lets the model package build special-token-safe prompt IDs
// without importing a concrete tokenizer implementation.
type EncodeTextFunc func(string) []int

func BuildTemplateChatPromptIDs(messages []TextChatMessage, specials SpecialTokenIDs, encode EncodeTextFunc, opts ChatRenderOptions) (PromptIDs, error) {
	if encode == nil {
		return PromptIDs{}, fmt.Errorf("nil DiffusionGemma text encoder")
	}
	ids := make([]int, 0)
	if opts.AddBOS {
		if specials.BOS < 0 {
			return PromptIDs{}, fmt.Errorf("DiffusionGemma BOS token ID unavailable")
		}
		ids = append(ids, specials.BOS)
	}
	start := 0
	if opts.EnableThinking || firstRoleIsSystem(messages) {
		if specials.BOT < 0 || specials.EOT < 0 {
			return PromptIDs{}, fmt.Errorf("DiffusionGemma turn token IDs unavailable")
		}
		ids = append(ids, specials.BOT)
		ids = append(ids, encode("system")...)
		ids = append(ids, 107) // \n token
		if opts.EnableThinking {
			if specials.THINK < 0 {
				return PromptIDs{}, fmt.Errorf("DiffusionGemma think token ID unavailable")
			}
			ids = append(ids, specials.THINK)
			ids = append(ids, 107) // \n token after <|think|>
		}
		if firstRoleIsSystem(messages) {
			ids = append(ids, encode(messages[0].Content)...)
			start = 1
		}
		ids = append(ids, specials.EOT)
		ids = append(ids, 107) // \n token after system turn end
	}
	for _, msg := range messages[start:] {
		if specials.BOT < 0 || specials.EOT < 0 {
			return PromptIDs{}, fmt.Errorf("DiffusionGemma turn token IDs unavailable")
		}
		role := normalizeTemplateRole(msg.Role)
		ids = append(ids, specials.BOT)
		ids = append(ids, encode(role)...)
		ids = append(ids, 107) // \n token
		ids = append(ids, encode(msg.Content)...)
		ids = append(ids, specials.EOT)
		ids = append(ids, 107) // \n token after turn end
	}
	if opts.AddGenerationPrompt {
		if specials.BOT < 0 {
			return PromptIDs{}, fmt.Errorf("DiffusionGemma begin-turn token ID unavailable")
		}
		ids = append(ids, specials.BOT)
		ids = append(ids, encode("model")...)
		ids = append(ids, 107) // \n token
		if !opts.EnableThinking {
			var err error
			ids, err = appendThoughtChannelPromptIDs(ids, specials, encode)
			if err != nil {
				return PromptIDs{}, err
			}
		}
	}
	return PromptIDs{InputIDs: ids}, nil
}

func appendThoughtChannelPromptIDs(ids []int, specials SpecialTokenIDs, encode EncodeTextFunc) ([]int, error) {
	if specials.BOC < 0 || specials.EOC < 0 {
		return nil, fmt.Errorf("DiffusionGemma channel token IDs unavailable")
	}
	ids = append(ids, specials.BOC)
	ids = append(ids, encode("thought")...)
	ids = append(ids, 107) // \n token
	ids = append(ids, specials.EOC)
	return ids, nil
}
