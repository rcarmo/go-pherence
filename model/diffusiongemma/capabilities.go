package diffusiongemma

// RuntimeCapabilities summarizes the current native DiffusionGemma runtime
// implementation state. It is intentionally explicit about remaining reference
// gaps so callers do not mistake text-only GGUF support for full multimodal
// reference completion.
type RuntimeCapabilities struct {
	Metadata                         bool `json:"metadata"`
	TensorInventory                  bool `json:"tensor_inventory"`
	TextTensorPlan                   bool `json:"text_tensor_plan"`
	ProcessorMetadata                bool `json:"processor_metadata"`
	TokenizerMetadata                bool `json:"tokenizer_metadata"`
	TextChatPrompt                   bool `json:"text_chat_prompt"`
	ImageProcessorPreprocess         bool `json:"image_processor_preprocess"`
	ImageSoftTokenPrompt             bool `json:"image_soft_token_prompt"`
	VisionTensorPlan                 bool `json:"vision_tensor_plan"`
	VisionForwardPlan                bool `json:"vision_forward_plan"`                  // semantic 27-layer graph/binding plan is available for safetensors vision tensors
	VisionTowerPrefix                bool `json:"vision_tower_prefix"`                  // bounded preloaded CPU prefix execution scaffold exists; full image-sequence tower remains pending
	VisionStreamingPrefix            bool `json:"vision_streaming_prefix"`              // streaming one-layer-at-a-time prefix/full-tower entrypoints exist; reference fixtures still pending
	VisionFullStreamingMaxPatches    int  `json:"vision_full_streaming_max_patches"`    // CPU scaffold safety limit; env override is explicit reference-validation intent
	VisionFullStreamingOverride      bool `json:"vision_full_streaming_override"`       // true when GO_PHERENCE_DIFFUSIONGEMMA_VISION_FULL_STREAMING_MAX_PATCHES is explicitly set
	VisionFullStreamingOverrideValid bool `json:"vision_full_streaming_override_valid"` // true when the explicit max-patch override parses successfully
	VisionEmbeddingBoundary          bool `json:"vision_embedding_boundary"`            // safetensors-side patch/projection/tower/insertion boundary; full vision reference fixtures still pending
	Sampler                          bool `json:"sampler"`
	CanvasEmbedding                  bool `json:"canvas_embedding"`
	SelfConditioning                 bool `json:"self_conditioning"`
	SelfConditioningFeedback         bool `json:"self_conditioning_feedback"`
	RMSNorm                          bool `json:"rms_norm"`
	DenseMLP                         bool `json:"dense_mlp"`
	Router                           bool `json:"router"`
	Experts                          bool `json:"experts"`
	LayerScalar                      bool `json:"layer_scalar"`
	FinalNorm                        bool `json:"final_norm"`
	LMHead                           bool `json:"lm_head"`

	SelfAttentionScaffold      bool                       `json:"self_attention_scaffold"`
	RoPE                       bool                       `json:"rope"`
	SlidingWindowMask          bool                       `json:"sliding_window_mask"`
	EncoderKVConcat            bool                       `json:"encoder_kv_concat"`
	TextOnlyScaffoldReady      bool                       `json:"text_only_scaffold_ready"`
	ReferenceComplete          bool                       `json:"reference_complete"`
	RuntimeReady               bool                       `json:"runtime_ready"`
	ImplementedOps             int                        `json:"implemented_ops"`
	ReferenceCompleteOps       int                        `json:"reference_complete_ops"`
	TotalOps                   int                        `json:"total_ops"`
	TextImplementedOps         int                        `json:"text_implemented_ops"`
	TextReferenceCompleteOps   int                        `json:"text_reference_complete_ops"`
	TextTotalOps               int                        `json:"text_total_ops"`
	VisionImplementedOps       int                        `json:"vision_implemented_ops"`
	VisionReferenceCompleteOps int                        `json:"vision_reference_complete_ops"`
	VisionTotalOps             int                        `json:"vision_total_ops"`
	OperationDomains           map[string]OpDomainSummary `json:"operation_domains,omitempty"`
	MissingForReference        []string                   `json:"missing_for_reference,omitempty"`
}

func Capabilities() RuntimeCapabilities {
	missing := MissingReferenceGaps()
	ops := OperationStatuses()
	implementedOps, referenceCompleteOps, totalOps := OperationStatusSummaryFromStatuses(ops)
	domains := OperationDomainSummaries(ops)
	text := domains["text"]
	vision := domains["vision"]
	visionOverride, visionOverrideValid := fullStreamingVisionPatchLimitOverrideState()
	return RuntimeCapabilities{
		Metadata:                         true,
		TensorInventory:                  true,
		TextTensorPlan:                   true,
		ProcessorMetadata:                true,
		TokenizerMetadata:                true,
		TextChatPrompt:                   true,
		ImageProcessorPreprocess:         true,
		ImageSoftTokenPrompt:             true,
		VisionTensorPlan:                 true,
		VisionForwardPlan:                true,
		VisionTowerPrefix:                true,
		VisionStreamingPrefix:            true,
		VisionFullStreamingMaxPatches:    MaxFullStreamingVisionPatches(),
		VisionFullStreamingOverride:      visionOverride,
		VisionFullStreamingOverrideValid: visionOverrideValid,
		VisionEmbeddingBoundary:          true,
		Sampler:                          true,
		CanvasEmbedding:                  true,
		SelfConditioning:                 true,
		SelfConditioningFeedback:         true,
		RMSNorm:                          true,
		DenseMLP:                         true,
		Router:                           true,
		Experts:                          true,
		LayerScalar:                      true,
		FinalNorm:                        true,
		LMHead:                           true,

		SelfAttentionScaffold:      true,
		RoPE:                       true,
		SlidingWindowMask:          true,
		EncoderKVConcat:            true,
		TextOnlyScaffoldReady:      true,
		ReferenceComplete:          false,
		RuntimeReady:               false,
		ImplementedOps:             implementedOps,
		ReferenceCompleteOps:       referenceCompleteOps,
		TotalOps:                   totalOps,
		TextImplementedOps:         text.Implemented,
		TextReferenceCompleteOps:   text.ReferenceComplete,
		TextTotalOps:               text.Total,
		VisionImplementedOps:       vision.Implemented,
		VisionReferenceCompleteOps: vision.ReferenceComplete,
		VisionTotalOps:             vision.Total,
		OperationDomains:           domains,
		MissingForReference:        missing,
	}
}
