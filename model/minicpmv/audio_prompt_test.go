package minicpmv

import (
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/config"
)

func TestBuildAudioPlaceholder(t *testing.T) {
	got, err := BuildAudioPlaceholder(3, "<ap>", "<as>", "<ae>", true)
	if err != nil {
		t.Fatalf("BuildAudioPlaceholder: %v", err)
	}
	if got != "<as><ap><ap><ap><ae>" {
		t.Fatalf("placeholder=%q", got)
	}
}

func TestBuildAudioPromptText(t *testing.T) {
	tok := &config.MiniCPMVTokenizerMetadata{AudioPatch: "<audio_patch>", AudioStart: "<audio_start>", AudioEnd: "<audio_end>"}
	prompt, err := BuildAudioPromptText("Transcribe this.", 2, 4, tok, PromptTextOptions{UserPrefix: "User: ", AssistantPrefix: "\nAssistant:"})
	if err != nil {
		t.Fatalf("BuildAudioPromptText: %v", err)
	}
	if prompt.Audios != 2 || prompt.PatchTokens != 4 || strings.Count(prompt.Text, "<audio_patch>") != 8 || strings.Count(prompt.Text, "<audio_start>") != 2 || !strings.Contains(prompt.Text, "Transcribe this.") {
		t.Fatalf("bad audio prompt: %+v", prompt)
	}
}

func TestBuildAudioPromptTextRejectsMissingPatchCount(t *testing.T) {
	if _, err := BuildAudioPromptText("x", 1, 0, nil, PromptTextOptions{}); err == nil {
		t.Fatalf("expected missing patch count error")
	}
}
