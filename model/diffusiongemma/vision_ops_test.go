package diffusiongemma

import "testing"

func TestBuildVisionForwardOpPlan(t *testing.T) {
	shape := Shape{VisionLayers: 2}
	plan := BuildVisionForwardOpPlan(shape, nil)
	if !plan.Ready || plan.Reason != "" {
		t.Fatalf("plan ready=%v reason=%q", plan.Ready, plan.Reason)
	}
	wantPrefix := []OpKind{OpImagePreprocess, OpImageSoftTokenPrompt, OpVisionPatchEmbedding}
	if len(plan.Prefix) != len(wantPrefix) {
		t.Fatalf("prefix len=%d want %d", len(plan.Prefix), len(wantPrefix))
	}
	for i := range wantPrefix {
		if plan.Prefix[i] != wantPrefix[i] {
			t.Fatalf("prefix[%d]=%s want %s", i, plan.Prefix[i], wantPrefix[i])
		}
	}
	wantLayer := []OpKind{OpVisionInputNorm, OpVisionSelfAttention, OpVisionPostAttention, OpVisionPreFFN, OpVisionDenseMLP, OpVisionPostFFN}
	if len(plan.Layers) != shape.VisionLayers*len(wantLayer) {
		t.Fatalf("layers len=%d want %d", len(plan.Layers), shape.VisionLayers*len(wantLayer))
	}
	for layer := 0; layer < shape.VisionLayers; layer++ {
		for j, want := range wantLayer {
			op := plan.Layers[layer*len(wantLayer)+j]
			if op.Layer != layer || op.Type != "vision" || op.Kind != want {
				t.Fatalf("layer op[%d,%d]=%+v want kind=%s", layer, j, op, want)
			}
		}
	}
	if len(plan.Tail) != 1 || plan.Tail[0] != OpVisionEmbeddingInsert {
		t.Fatalf("tail=%v", plan.Tail)
	}
}

func TestBuildVisionForwardOpPlanRequiresReadyBindings(t *testing.T) {
	bad := VisionForwardPlan{Ready: false, Missing: []string{"vision layer 0 q_proj"}}
	plan := BuildVisionForwardOpPlan(Shape{VisionLayers: 1}, &bad)
	if plan.Ready || plan.Reason != "vision forward bindings incomplete" {
		t.Fatalf("plan ready=%v reason=%q", plan.Ready, plan.Reason)
	}
}

func TestLocalDiffusionGemmaVisionForwardOpPlan(t *testing.T) {
	dir := localDiffusionGemmaModelDir(t)
	meta, err := LoadMetadata(dir)
	if err != nil {
		t.Fatal(err)
	}
	w, err := OpenVisionWeights(dir, meta.Shape)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	fp := w.ForwardPlan()
	plan := BuildVisionForwardOpPlan(meta.Shape, &fp)
	if !plan.Ready {
		t.Fatalf("vision op plan not ready: %q", plan.Reason)
	}
	if got, want := len(plan.Layers), meta.Shape.VisionLayers*6; got != want {
		t.Fatalf("vision op layers=%d want %d", got, want)
	}
}
