package minicpmv

const (
	RuntimeStatusScaffold = "metadata_scaffold_ready"
	RuntimeStatusPending  = "tensor_execution_pending"
)

type Capabilities struct {
	RuntimeStatus              string `json:"runtime_status"`
	ConfigParsing              bool   `json:"config_parsing"`
	ProcessorMetadata          bool   `json:"processor_metadata"`
	TokenizerMetadata          bool   `json:"tokenizer_metadata"`
	GenerationMetadata         bool   `json:"generation_metadata"`
	ChatTemplateSummary        bool   `json:"chat_template_summary"`
	ImageSpecialTokens         bool   `json:"image_special_tokens"`
	AudioSpecialTokens         bool   `json:"audio_special_tokens"`
	ImagePromptPlanning        bool   `json:"image_prompt_planning"`
	AudioPromptPlanning        bool   `json:"audio_prompt_planning"`
	MultimodalPromptPlanning   bool   `json:"multimodal_prompt_planning"`
	ImagePreprocessing         bool   `json:"image_preprocessing"`
	ImageFileInspection        bool   `json:"image_file_inspection"`
	SliceModePlanning          bool   `json:"slice_mode_planning"`
	TensorInventory            bool   `json:"tensor_inventory"`
	TensorShapeValidation      bool   `json:"tensor_shape_validation"`
	ExplicitSafetensorsPath    bool   `json:"explicit_safetensors_path"`
	TextExecutionPlan          bool   `json:"text_execution_plan"`
	VisionExecutionPlan        bool   `json:"vision_execution_plan"`
	ResamplerTensorPlan        bool   `json:"resampler_tensor_plan"`
	AudioExecutionPlan         bool   `json:"audio_execution_plan"`
	RuntimeInterfaces          bool   `json:"runtime_interfaces"`
	ReadinessReport            bool   `json:"readiness_report"`
	EmbeddingInjectionBoundary bool   `json:"embedding_injection_boundary"`
	InspectorReadinessGates    bool   `json:"inspector_readiness_gates"`
	ValidationGate             bool   `json:"validation_gate"`

	TextRuntimeGeneration bool     `json:"text_runtime_generation"`
	VisionTowerRuntime    bool     `json:"vision_tower_runtime"`
	ResamplerRuntime      bool     `json:"resampler_runtime"`
	AudioEncoderRuntime   bool     `json:"audio_encoder_runtime"`
	EndToEndGeneration    bool     `json:"end_to_end_generation"`
	PendingRuntimeSteps   []string `json:"pending_runtime_steps,omitempty"`
}

func CurrentCapabilities() Capabilities {
	return Capabilities{
		RuntimeStatus:              RuntimeStatusPending,
		ConfigParsing:              true,
		ProcessorMetadata:          true,
		TokenizerMetadata:          true,
		GenerationMetadata:         true,
		ChatTemplateSummary:        true,
		ImageSpecialTokens:         true,
		AudioSpecialTokens:         true,
		ImagePromptPlanning:        true,
		AudioPromptPlanning:        true,
		MultimodalPromptPlanning:   true,
		ImagePreprocessing:         true,
		ImageFileInspection:        true,
		SliceModePlanning:          true,
		TensorInventory:            true,
		TensorShapeValidation:      true,
		ExplicitSafetensorsPath:    true,
		TextExecutionPlan:          true,
		VisionExecutionPlan:        true,
		ResamplerTensorPlan:        true,
		AudioExecutionPlan:         true,
		RuntimeInterfaces:          true,
		ReadinessReport:            true,
		EmbeddingInjectionBoundary: true,
		InspectorReadinessGates:    true,
		ValidationGate:             true,
		PendingRuntimeSteps:        PendingRuntimeSteps(),
	}
}
