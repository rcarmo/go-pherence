package diffusiongemma

import "strings"

// TextChatMessage is a minimal text-only chat message for native scaffold
// rendering. Tool calls, multimodal parts, and reasoning channels remain future
// work; this follows the basic Gemma template shape for system/user/model turns.
type TextChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRenderOptions struct {
	AddBOS              bool `json:"add_bos"`
	EnableThinking      bool `json:"enable_thinking"`
	AddGenerationPrompt bool `json:"add_generation_prompt"`
}

func RenderSimpleChatTemplate(messages []TextChatMessage, proc *ProcessorMetadata, opts ChatRenderOptions) string {
	var b strings.Builder
	if opts.AddBOS && proc != nil && proc.BOS != "" {
		b.WriteString(proc.BOS)
	}
	start := 0
	if opts.EnableThinking || firstRoleIsSystem(messages) {
		writeTurnStart(&b, proc, "system")
		if opts.EnableThinking && proc != nil && proc.Think != "" {
			b.WriteString(proc.Think)
			b.WriteByte('\n')
		}
		if len(messages) > 0 && isSystemRole(messages[0].Role) {
			b.WriteString(strings.TrimSpace(messages[0].Content))
			start = 1
		}
		writeTurnEnd(&b, proc)
		b.WriteByte('\n')
	}
	for _, msg := range messages[start:] {
		role := normalizeTemplateRole(msg.Role)
		writeTurnStart(&b, proc, role)
		b.WriteString(strings.TrimSpace(msg.Content))
		writeTurnEnd(&b, proc)
		b.WriteByte('\n')
	}
	if opts.AddGenerationPrompt {
		writeTurnStart(&b, proc, "model")
	}
	return b.String()
}

func writeTurnStart(b *strings.Builder, proc *ProcessorMetadata, role string) {
	if proc != nil && proc.BOT != "" {
		b.WriteString(proc.BOT)
	} else {
		b.WriteString("<|turn>")
	}
	b.WriteString(role)
	b.WriteByte('\n')
}

func writeTurnEnd(b *strings.Builder, proc *ProcessorMetadata) {
	if proc != nil && proc.EOT != "" {
		b.WriteString(proc.EOT)
	} else {
		b.WriteString("<turn|>")
	}
	b.WriteByte('\n')
}

func firstRoleIsSystem(messages []TextChatMessage) bool {
	return len(messages) > 0 && isSystemRole(messages[0].Role)
}

func isSystemRole(role string) bool {
	role = strings.TrimSpace(role)
	return role == "system" || role == "developer"
}

func normalizeTemplateRole(role string) string {
	role = strings.TrimSpace(role)
	if role == "assistant" {
		return "model"
	}
	if role == "developer" {
		return "system"
	}
	if role == "" {
		return "user"
	}
	return role
}
