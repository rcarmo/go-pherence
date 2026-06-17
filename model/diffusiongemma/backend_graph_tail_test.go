package diffusiongemma

import (
	"strings"
	"testing"
)

func TestGGUFBackendGraphRejectsHostVisibleLMHead(t *testing.T) {
	t.Setenv("GO_PHERENCE_DIFFUSIONGEMMA_REQUIRE_GROUPED_EXPERT_GRAPH", "1")
	d := GPUDispatcher{GGUFExpertIndex: &GGUFExpertIndex{HiddenSize: 2}}
	weights := &TextWeights{}
	ops := ForwardOpPlan{Ready: true, Tail: []OpKind{OpLMHead}}
	bufs := ForwardBufferPlan{HiddenSize: 2, Hidden: 2, VocabSize: 4, CanvasLength: 1}
	_, err := d.RunTextForward(ForwardContext{Canvas: []int{1}, Step: 1}, weights, ops, bufs)
	if err == nil || !strings.Contains(err.Error(), "device-resident LM-head/self-conditioning") {
		t.Fatalf("error=%v, want device-resident LM-head/self-conditioning rejection", err)
	}
}
