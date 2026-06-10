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
		ids = append(ids, encode("system\n")...)
		if opts.EnableThinking {
			if specials.THINK < 0 {
				return PromptIDs{}, fmt.Errorf("DiffusionGemma think token ID unavailable")
			}
			ids = append(ids, specials.THINK)
			ids = append(ids, encode("\n")...)
		}
		if firstRoleIsSystem(messages) {
			ids = append(ids, encode(messages[0].Content)...)
			start = 1
		}
		ids = append(ids, specials.EOT)
	}
	for _, msg := range messages[start:] {
		if specials.BOT < 0 || specials.EOT < 0 {
			return PromptIDs{}, fmt.Errorf("DiffusionGemma turn token IDs unavailable")
		}
		role := normalizeTemplateRole(msg.Role)
		ids = append(ids, specials.BOT)
		ids = append(ids, encode(role+"\n"+msg.Content)...)
		ids = append(ids, specials.EOT)
	}
	if opts.AddGenerationPrompt {
		if specials.BOT < 0 {
			return PromptIDs{}, fmt.Errorf("DiffusionGemma begin-turn token ID unavailable")
		}
		ids = append(ids, specials.BOT)
		ids = append(ids, encode("model\n")...)
	}
	return PromptIDs{InputIDs: ids}, nil
}
