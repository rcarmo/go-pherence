package diffusiongemma

// VisionForwardOpPlan is the semantic graph skeleton for the Gemma4 image path.
// It deliberately separates graph planning from execution: current diagnostics
// can prove the tower has complete bindings without claiming reference-correct
// full vision execution.
type VisionForwardOpPlan struct {
	Prefix []OpKind  `json:"prefix"`
	Layers []LayerOp `json:"layers"`
	Tail   []OpKind  `json:"tail"`
	Ready  bool      `json:"ready"`
	Reason string    `json:"reason,omitempty"`
}

func BuildVisionForwardOpPlan(shape Shape, visionPlan *VisionForwardPlan) VisionForwardOpPlan {
	if shape.VisionLayers <= 0 {
		return VisionForwardOpPlan{Ready: false, Reason: "missing vision layers"}
	}
	if visionPlan != nil && !visionPlan.Ready {
		return VisionForwardOpPlan{Ready: false, Reason: "vision forward bindings incomplete"}
	}
	plan := VisionForwardOpPlan{
		Ready:  true,
		Prefix: []OpKind{OpImagePreprocess, OpImageSoftTokenPrompt, OpVisionPatchEmbedding},
		Tail:   []OpKind{OpVisionEmbeddingInsert},
	}
	for i := 0; i < shape.VisionLayers; i++ {
		plan.Layers = append(plan.Layers,
			LayerOp{Layer: i, Type: "vision", Kind: OpVisionInputNorm},
			LayerOp{Layer: i, Type: "vision", Kind: OpVisionSelfAttention},
			LayerOp{Layer: i, Type: "vision", Kind: OpVisionPostAttention},
			LayerOp{Layer: i, Type: "vision", Kind: OpVisionPreFFN},
			LayerOp{Layer: i, Type: "vision", Kind: OpVisionDenseMLP},
			LayerOp{Layer: i, Type: "vision", Kind: OpVisionPostFFN},
		)
	}
	return plan
}
