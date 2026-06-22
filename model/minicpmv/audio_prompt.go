package minicpmv

import (
	"fmt"
	"strings"

	"github.com/rcarmo/go-pherence/loader/config"
)

const (
	DefaultAudioToken      = "<audio>"
	DefaultAudioPatchToken = "<audio_patch>"
	DefaultAudioStartToken = "<audio_start>"
	DefaultAudioEndToken   = "<audio_end>"
)

type AudioPromptText struct {
	Text             string `json:"text"`
	AudioPlaceholder string `json:"audio_placeholder"`
	Audios           int    `json:"audios"`
	PatchTokens      int    `json:"patch_tokens"`
	UseStartEnd      bool   `json:"use_start_end"`
}

func BuildAudioPlaceholder(patchTokens int, audioPatchToken, audioStartToken, audioEndToken string, useStartEnd bool) (string, error) {
	if patchTokens <= 0 {
		return "", fmt.Errorf("MiniCPM-O audio placeholder: patch token count must be positive, got %d", patchTokens)
	}
	if audioPatchToken == "" {
		audioPatchToken = DefaultAudioPatchToken
	}
	patches := strings.Repeat(audioPatchToken, patchTokens)
	if !useStartEnd {
		return patches, nil
	}
	if audioStartToken == "" {
		audioStartToken = DefaultAudioStartToken
	}
	if audioEndToken == "" {
		audioEndToken = DefaultAudioEndToken
	}
	return audioStartToken + patches + audioEndToken, nil
}

func BuildAudioPromptText(question string, audios, patchTokens int, tok *config.MiniCPMVTokenizerMetadata, opts PromptTextOptions) (AudioPromptText, error) {
	out := AudioPromptText{Audios: audios, PatchTokens: patchTokens, UseStartEnd: true}
	if audios < 0 {
		return out, fmt.Errorf("MiniCPM-O audio prompt text: audios must be non-negative, got %d", audios)
	}
	patch, start, end := DefaultAudioPatchToken, DefaultAudioStartToken, DefaultAudioEndToken
	if tok != nil {
		patch = firstNonEmpty(tok.AudioPatch, patch)
		start = firstNonEmpty(tok.AudioStart, start)
		end = firstNonEmpty(tok.AudioEnd, end)
	}
	placeholder, err := BuildAudioPlaceholder(patchTokens, patch, start, end, true)
	if err != nil {
		return out, err
	}
	out.AudioPlaceholder = placeholder
	sep := opts.Separator
	if sep == "" {
		sep = "\n"
	}
	var b strings.Builder
	for i := 0; i < audios; i++ {
		if i > 0 {
			b.WriteString(sep)
		}
		b.WriteString(placeholder)
	}
	if question != "" {
		if b.Len() > 0 {
			b.WriteString(sep)
		}
		b.WriteString(question)
	}
	text := b.String()
	if opts.UserPrefix != "" {
		text = opts.UserPrefix + text
	}
	if opts.AssistantPrefix != "" {
		text += opts.AssistantPrefix
	}
	out.Text = text
	return out, nil
}
