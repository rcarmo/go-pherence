package diffusiongemma

import "testing"

func TestDiffusionGemmaTextForwardOpPlanMatchesLlamaCppOrder(t *testing.T) {
	shape := Shape{TextLayers: 2}
	plan := BuildForwardOpPlan(shape, nil)
	if !plan.Ready {
		t.Fatalf("plan not ready: %s", plan.Reason)
	}
	wantPrefix := []OpKind{OpCanvasEmbedding, OpSelfCondition}
	if !sameOpKinds(plan.Prefix, wantPrefix) {
		t.Fatalf("prefix ops=%v want %v", plan.Prefix, wantPrefix)
	}
	wantLayer := []OpKind{
		OpInputNorm,     // input RMSNorm before QKV
		OpSelfAttention, // q/k/v projection, q/k norm, RoPE, KV cache/concat, attention + O projection
		OpPostAttention, // attn_post_norm + residual -> attn_out
		OpPreMoE,        // ffn_pre_norm_2 / MoE input norm on attn_out
		OpDenseMLP,      // shared dense Gemma4 MLP from attn_out
		OpRouter,        // router no-scale norm, scale, gate projection, softmax/top-k
		OpExperts,       // selected-expert grouped mul_mat_id-style gate/up/down + weighting/aggregation
		OpPostMoE,       // ffn_post_norm over dense+MoE branch + residual
		OpLayerScalar,   // encoder/canvas region-aware scalar
	}
	for layer := 0; layer < shape.TextLayers; layer++ {
		start := layer * len(wantLayer)
		got := make([]OpKind, len(wantLayer))
		for i := range wantLayer {
			got[i] = plan.Layers[start+i].Kind
			if plan.Layers[start+i].Layer != layer {
				t.Fatalf("layer op[%d] layer=%d want %d", start+i, plan.Layers[start+i].Layer, layer)
			}
		}
		if !sameOpKinds(got, wantLayer) {
			t.Fatalf("layer %d ops=%v want %v", layer, got, wantLayer)
		}
	}
	wantTail := []OpKind{OpFinalNorm, OpLMHead}
	if !sameOpKinds(plan.Tail, wantTail) {
		t.Fatalf("tail ops=%v want %v", plan.Tail, wantTail)
	}
}

func sameOpKinds(a, b []OpKind) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
