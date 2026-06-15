package diffusiongemma

import "testing"

func TestBuildForwardOpPlanTextGraphContract(t *testing.T) {
	shape := Shape{TextLayers: 30, LayerTypes: make([]string, 30)}
	for i := range shape.LayerTypes {
		shape.LayerTypes[i] = "sliding_attention"
	}
	plan := BuildForwardOpPlan(shape, nil)
	if !plan.Ready || plan.Reason != "" {
		t.Fatalf("plan ready=%v reason=%q", plan.Ready, plan.Reason)
	}
	if len(plan.Prefix) != 2 || plan.Prefix[0] != OpCanvasEmbedding || plan.Prefix[1] != OpSelfCondition {
		t.Fatalf("prefix=%v", plan.Prefix)
	}
	if len(plan.Layers) != shape.TextLayers*9 {
		t.Fatalf("layer ops=%d want %d", len(plan.Layers), shape.TextLayers*9)
	}
	wantKinds := []OpKind{OpInputNorm, OpSelfAttention, OpPostAttention, OpPreMoE, OpDenseMLP, OpRouter, OpExperts, OpPostMoE, OpLayerScalar}
	for layer := 0; layer < shape.TextLayers; layer++ {
		for i, want := range wantKinds {
			op := plan.Layers[layer*len(wantKinds)+i]
			if op.Layer != layer || op.Kind != want {
				t.Fatalf("layer=%d op=%d got=%+v want kind=%s", layer, i, op, want)
			}
		}
	}
	if len(plan.Tail) != 2 || plan.Tail[0] != OpFinalNorm || plan.Tail[1] != OpLMHead {
		t.Fatalf("tail=%v", plan.Tail)
	}
}

func TestBuildForwardOpPlanRejectsMissingTextLayers(t *testing.T) {
	plan := BuildForwardOpPlan(Shape{}, nil)
	if plan.Ready || plan.Reason != "missing text layers" {
		t.Fatalf("plan ready=%v reason=%q", plan.Ready, plan.Reason)
	}
}

func TestBuildForwardOpPlanRejectsIncompleteTextBindings(t *testing.T) {
	textPlan := TextForwardPlan{Ready: false, Missing: []string{"layer 0 q_proj"}}
	plan := BuildForwardOpPlan(Shape{TextLayers: 1}, &textPlan)
	if plan.Ready || plan.Reason != "text forward bindings incomplete" {
		t.Fatalf("plan ready=%v reason=%q", plan.Ready, plan.Reason)
	}
}
