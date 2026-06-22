package minicpmv

import (
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/config"
)

func TestBuildImagePlaceholderStartEnd(t *testing.T) {
	got, err := BuildImagePlaceholder(3, "<p>", "<s>", "<e>", true)
	if err != nil {
		t.Fatalf("BuildImagePlaceholder: %v", err)
	}
	if got != "<s><p><p><p><e>" {
		t.Fatalf("placeholder=%q", got)
	}
}

func TestBuildImagePlaceholderPatchOnly(t *testing.T) {
	got, err := BuildImagePlaceholder(2, "<p>", "", "", false)
	if err != nil {
		t.Fatalf("BuildImagePlaceholder: %v", err)
	}
	if got != "<p><p>" {
		t.Fatalf("placeholder=%q", got)
	}
}

func TestBuildPromptText(t *testing.T) {
	summary := config.MiniCPMVSummary{NumQuery: 4, UseImageStartEnd: true}
	tok := &config.MiniCPMVTokenizerMetadata{ImagePatch: "<im_patch>", ImageStart: "<im_start>", ImageEnd: "<im_end>"}
	prompt, err := BuildPromptText("What is in the image?", 2, summary, tok, PromptTextOptions{UserPrefix: "User: ", AssistantPrefix: "\nAssistant:"})
	if err != nil {
		t.Fatalf("BuildPromptText: %v", err)
	}
	if prompt.Images != 2 || prompt.PatchTokens != 4 || !prompt.UseStartEnd {
		t.Fatalf("bad metadata: %+v", prompt)
	}
	if strings.Count(prompt.Text, "<im_patch>") != 8 || strings.Count(prompt.Text, "<im_start>") != 2 || !strings.Contains(prompt.Text, "What is in the image?") || !strings.HasPrefix(prompt.Text, "User: ") || !strings.HasSuffix(prompt.Text, "Assistant:") {
		t.Fatalf("bad prompt text: %q", prompt.Text)
	}
}

func TestBuildPromptTextRejectsMissingQuery(t *testing.T) {
	_, err := BuildPromptText("x", 1, config.MiniCPMVSummary{}, nil, PromptTextOptions{})
	if err == nil {
		t.Fatalf("expected missing num_query to fail")
	}
}
