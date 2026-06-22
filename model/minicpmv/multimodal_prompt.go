package minicpmv

import (
	"github.com/rcarmo/go-pherence/loader/config"
)

type MultiModalPromptOptions struct {
	Question         string
	Images           int
	Audios           int
	AudioPatchTokens int
	PromptOptions    PromptTextOptions
}

type MultiModalPromptPlan struct {
	Text        string           `json:"text"`
	ImagePrompt *PromptText      `json:"image_prompt,omitempty"`
	AudioPrompt *AudioPromptText `json:"audio_prompt,omitempty"`
	Images      int              `json:"images"`
	Audios      int              `json:"audios"`
}

func BuildMultiModalPromptPlan(summary config.MiniCPMVSummary, tok *config.MiniCPMVTokenizerMetadata, opts MultiModalPromptOptions) (MultiModalPromptPlan, error) {
	out := MultiModalPromptPlan{Images: opts.Images, Audios: opts.Audios}
	sep := opts.PromptOptions.Separator
	if sep == "" {
		sep = "\n"
	}
	var text string
	if opts.Images > 0 {
		p, err := BuildPromptText("", opts.Images, summary, tok, opts.PromptOptions)
		if err != nil {
			return out, err
		}
		out.ImagePrompt = &p
		text = p.Text
	}
	if opts.Audios > 0 {
		patches := opts.AudioPatchTokens
		if patches <= 0 {
			patches = summary.NumQuery
		}
		p, err := BuildAudioPromptText("", opts.Audios, patches, tok, PromptTextOptions{Separator: sep})
		if err != nil {
			return out, err
		}
		out.AudioPrompt = &p
		if text != "" {
			text += sep
		}
		text += p.Text
	}
	if opts.Question != "" {
		if text != "" {
			text += sep
		}
		text += opts.Question
	}
	if opts.PromptOptions.UserPrefix != "" {
		text = opts.PromptOptions.UserPrefix + text
	}
	if opts.PromptOptions.AssistantPrefix != "" {
		text += opts.PromptOptions.AssistantPrefix
	}
	out.Text = text
	return out, nil
}
