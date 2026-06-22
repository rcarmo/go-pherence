package minicpmv

import (
	"fmt"
	"strings"

	"github.com/rcarmo/go-pherence/loader/config"
)

type PromptTextOptions struct {
	UserPrefix      string `json:"user_prefix,omitempty"`
	AssistantPrefix string `json:"assistant_prefix,omitempty"`
	Separator       string `json:"separator,omitempty"`
}

type PromptText struct {
	Text             string `json:"text"`
	ImagePlaceholder string `json:"image_placeholder"`
	Images           int    `json:"images"`
	PatchTokens      int    `json:"patch_tokens"`
	UseStartEnd      bool   `json:"use_start_end"`
}

// BuildImagePlaceholder returns the textual MiniCPM image placeholder sequence
// that should tokenize into one image span. It does not run a tokenizer; it
// locks the model-family convention before tokenizer integration/runtime decode.
func BuildImagePlaceholder(numQuery int, imagePatchToken, imageStartToken, imageEndToken string, useStartEnd bool) (string, error) {
	if numQuery <= 0 {
		return "", fmt.Errorf("MiniCPM-V/O image placeholder: num_query must be positive, got %d", numQuery)
	}
	if imagePatchToken == "" {
		imagePatchToken = DefaultImagePatchToken
	}
	patches := strings.Repeat(imagePatchToken, numQuery)
	if !useStartEnd {
		return patches, nil
	}
	if imageStartToken == "" {
		imageStartToken = DefaultImageStartToken
	}
	if imageEndToken == "" {
		imageEndToken = DefaultImageEndToken
	}
	return imageStartToken + patches + imageEndToken, nil
}

func BuildPromptText(question string, images int, summary config.MiniCPMVSummary, tok *config.MiniCPMVTokenizerMetadata, opts PromptTextOptions) (PromptText, error) {
	out := PromptText{Images: images, PatchTokens: summary.NumQuery, UseStartEnd: summary.UseImageStartEnd}
	if images < 0 {
		return out, fmt.Errorf("MiniCPM-V/O prompt text: images must be non-negative, got %d", images)
	}
	patch, start, end := DefaultImagePatchToken, DefaultImageStartToken, DefaultImageEndToken
	if tok != nil {
		patch = firstNonEmpty(tok.ImagePatch, patch)
		start = firstNonEmpty(tok.ImageStart, start)
		end = firstNonEmpty(tok.ImageEnd, end)
	}
	placeholder, err := BuildImagePlaceholder(summary.NumQuery, patch, start, end, summary.UseImageStartEnd)
	if err != nil {
		return out, err
	}
	out.ImagePlaceholder = placeholder
	sep := opts.Separator
	if sep == "" {
		sep = "\n"
	}
	var b strings.Builder
	for i := 0; i < images; i++ {
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
		text = text + opts.AssistantPrefix
	}
	out.Text = text
	return out, nil
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}
