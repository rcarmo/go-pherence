package diffusiongemma

import (
	"strings"
	"testing"
)

func TestCPUDispatcherRunTextForwardReferencePathValidatesInputs(t *testing.T) {
	_, err := (CPUDispatcher{}).RunTextForward(ForwardContext{Canvas: []int{1}}, nil, ForwardOpPlan{Ready: true}, ForwardBufferPlan{})
	if err == nil || !strings.Contains(err.Error(), "missing text weights") {
		t.Fatalf("expected CPUDispatcher input validation error, got %v", err)
	}
}

func TestCPUDispatcherEncodePromptReferencePathValidatesInputs(t *testing.T) {
	_, err := (CPUDispatcher{}).EncodePrompt([]int{1}, nil, ForwardOpPlan{}, ForwardBufferPlan{})
	if err == nil || !strings.Contains(err.Error(), "encoder missing weights") {
		t.Fatalf("expected CPU prompt encoder input validation error, got %v", err)
	}
}

func TestCPUDispatcherEncodePromptSuffixGGUFReferencePathValidatesInputs(t *testing.T) {
	_, err := (CPUDispatcher{}).EncodePromptSuffixGGUF([]int{1}, nil, nil, ForwardOpPlan{}, ForwardBufferPlan{})
	if err == nil || !strings.Contains(err.Error(), "requires GGUFExpertIndex") {
		t.Fatalf("expected CPU suffix encoder input validation error, got %v", err)
	}
}
