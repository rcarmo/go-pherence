package diffusiongemma

// RuntimeCapabilities summarizes the current native DiffusionGemma runtime
// implementation state. It is intentionally explicit about partial attention
// support so callers do not mistake the scaffold for reference-complete
// inference.
type RuntimeCapabilities struct {
	Metadata              bool     `json:"metadata"`
	TensorInventory       bool     `json:"tensor_inventory"`
	TextTensorPlan        bool     `json:"text_tensor_plan"`
	Sampler               bool     `json:"sampler"`
	CanvasEmbedding       bool     `json:"canvas_embedding"`
	SelfConditioning      bool     `json:"self_conditioning"`
	RMSNorm               bool     `json:"rms_norm"`
	DenseMLP              bool     `json:"dense_mlp"`
	Router                bool     `json:"router"`
	Experts               bool     `json:"experts"`
	LayerScalar           bool     `json:"layer_scalar"`
	FinalNorm             bool     `json:"final_norm"`
	LMHead                bool     `json:"lm_head"`
	SelfAttentionScaffold bool     `json:"self_attention_scaffold"`
	RoPE                  bool     `json:"rope"`
	SlidingWindowMask     bool     `json:"sliding_window_mask"`
	EncoderKVConcat       bool     `json:"encoder_kv_concat"`
	ReferenceComplete     bool     `json:"reference_complete"`
	RuntimeReady          bool     `json:"runtime_ready"`
	MissingForReference   []string `json:"missing_for_reference,omitempty"`
}

func Capabilities() RuntimeCapabilities {
	missing := []string{"reference parity fixtures", "memory-efficient LM head", "vision/token processor integration"}
	return RuntimeCapabilities{
		Metadata:              true,
		TensorInventory:       true,
		TextTensorPlan:        true,
		Sampler:               true,
		CanvasEmbedding:       true,
		SelfConditioning:      true,
		RMSNorm:               true,
		DenseMLP:              true,
		Router:                true,
		Experts:               true,
		LayerScalar:           true,
		FinalNorm:             true,
		LMHead:                true,
		SelfAttentionScaffold: true,
		RoPE:                  true,
		SlidingWindowMask:     true,
		EncoderKVConcat:       true,
		ReferenceComplete:     false,
		RuntimeReady:          false,
		MissingForReference:   missing,
	}
}
