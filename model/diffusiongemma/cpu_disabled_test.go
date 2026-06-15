package diffusiongemma

import (
	"strings"
	"testing"
)

func TestCPUDispatcherRunTextForwardDisabled(t *testing.T) {
	_, err := (CPUDispatcher{}).RunTextForward(ForwardContext{Canvas: []int{1}}, nil, ForwardOpPlan{Ready: true}, ForwardBufferPlan{})
	if err == nil || !strings.Contains(err.Error(), "CPUDispatcher is disabled") {
		t.Fatalf("expected CPUDispatcher disabled error, got %v", err)
	}
}

func TestCPUDispatcherEncodePromptDisabled(t *testing.T) {
	_, err := (CPUDispatcher{}).EncodePrompt([]int{1}, nil, ForwardOpPlan{}, ForwardBufferPlan{})
	if err == nil || !strings.Contains(err.Error(), "CPU prompt encoder is disabled") {
		t.Fatalf("expected CPU prompt encoder disabled error, got %v", err)
	}
}

func TestCPUDispatcherEncodePromptSuffixGGUFDisabled(t *testing.T) {
	_, err := (CPUDispatcher{}).EncodePromptSuffixGGUF([]int{1}, nil, nil, ForwardOpPlan{}, ForwardBufferPlan{})
	if err == nil || !strings.Contains(err.Error(), "CPU GGUF suffix encoder is disabled") {
		t.Fatalf("expected CPU suffix encoder disabled error, got %v", err)
	}
}
