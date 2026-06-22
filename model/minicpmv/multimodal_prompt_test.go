package minicpmv

import (
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/config"
)

func TestBuildMultiModalPromptPlan(t *testing.T) {
	summary := config.MiniCPMVSummary{NumQuery: 2, UseImageStartEnd: true}
	tok := &config.MiniCPMVTokenizerMetadata{ImagePatch: "<im_patch>", ImageStart: "<im_start>", ImageEnd: "<im_end>", AudioPatch: "<audio_patch>", AudioStart: "<audio_start>", AudioEnd: "<audio_end>"}
	plan, err := BuildMultiModalPromptPlan(summary, tok, MultiModalPromptOptions{Question: "Describe both.", Images: 1, Audios: 1, PromptOptions: PromptTextOptions{UserPrefix: "User: ", AssistantPrefix: "\nAssistant:"}})
	if err != nil {
		t.Fatalf("BuildMultiModalPromptPlan: %v", err)
	}
	if plan.ImagePrompt == nil || plan.AudioPrompt == nil || plan.Images != 1 || plan.Audios != 1 {
		t.Fatalf("bad multimodal metadata: %+v", plan)
	}
	if strings.Count(plan.Text, "<im_patch>") != 2 || strings.Count(plan.Text, "<audio_patch>") != 2 || !strings.Contains(plan.Text, "Describe both.") || !strings.HasPrefix(plan.Text, "User: ") || !strings.HasSuffix(plan.Text, "Assistant:") {
		t.Fatalf("bad multimodal text: %q", plan.Text)
	}
}

func TestBuildMultiModalPromptPlanAudioPatchOverride(t *testing.T) {
	summary := config.MiniCPMVSummary{NumQuery: 2, UseImageStartEnd: true}
	plan, err := BuildMultiModalPromptPlan(summary, nil, MultiModalPromptOptions{Audios: 1, AudioPatchTokens: 3})
	if err != nil {
		t.Fatalf("BuildMultiModalPromptPlan: %v", err)
	}
	if plan.AudioPrompt == nil || strings.Count(plan.Text, DefaultAudioPatchToken) != 3 {
		t.Fatalf("bad audio override: %+v text=%q", plan.AudioPrompt, plan.Text)
	}
}
