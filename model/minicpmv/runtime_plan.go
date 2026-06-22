package minicpmv

import (
	"fmt"
	"strings"

	"github.com/rcarmo/go-pherence/loader/config"
)

type RuntimeOpStatus struct {
	Name   string `json:"name"`
	Ready  bool   `json:"ready"`
	Reason string `json:"reason,omitempty"`
}

type RuntimePlan struct {
	ConfigReady          bool              `json:"config_ready"`
	ProcessorReady       bool              `json:"processor_ready"`
	TokenizerReady       bool              `json:"tokenizer_ready"`
	SpecialTokensReady   bool              `json:"special_tokens_ready"`
	TensorMetadataReady  bool              `json:"tensor_metadata_ready"`
	ImagePreprocessReady bool              `json:"image_preprocess_ready"`
	PromptPlanningReady  bool              `json:"prompt_planning_ready"`
	RuntimeReady         bool              `json:"runtime_ready"`
	Ops                  []RuntimeOpStatus `json:"ops"`
}

func BuildRuntimePlan(summary config.MiniCPMVSummary, processor *config.MiniCPMVProcessorConfig, tokenizer *config.MiniCPMVTokenizerMetadata, tensors *TensorInventory) RuntimePlan {
	plan := RuntimePlan{}
	add := func(name string, ready bool, reason string) {
		plan.Ops = append(plan.Ops, RuntimeOpStatus{Name: name, Ready: ready, Reason: reason})
	}
	plan.ConfigReady = summary.HiddenSize > 0 && summary.Layers > 0 && summary.Heads > 0 && summary.NumQuery > 0 && summary.ResamplerGrid*summary.ResamplerGrid == summary.NumQuery
	add("config", plan.ConfigReady, reasonIf(!plan.ConfigReady, "missing text or square resampler dimensions"))
	plan.ProcessorReady = processor != nil && (processor.NormalizedSize > 0 || summary.ImageSize > 0) && (processor.PatchSize > 0 || summary.PatchSize > 0)
	add("processor", plan.ProcessorReady, reasonIf(!plan.ProcessorReady, "missing image processor size/patch metadata"))
	plan.TokenizerReady = tokenizer != nil
	add("tokenizer", plan.TokenizerReady, reasonIf(!plan.TokenizerReady, "missing tokenizer sidecar metadata"))
	if _, err := ResolveSpecialTokenIDs(summary, tokenizer); err == nil {
		plan.SpecialTokensReady = true
	}
	add("special_tokens", plan.SpecialTokensReady, reasonIf(!plan.SpecialTokensReady, "missing image patch/start/end token ids"))
	plan.ImagePreprocessReady = plan.ConfigReady && (summary.ImageSize > 0 || processor != nil) && firstPositive(summary.PatchSize, processorPatch(processor)) > 0
	add("image_preprocess", plan.ImagePreprocessReady, reasonIf(!plan.ImagePreprocessReady, "missing image size or patch size"))
	plan.PromptPlanningReady = plan.ConfigReady && plan.SpecialTokensReady
	add("prompt_planning", plan.PromptPlanningReady, reasonIf(!plan.PromptPlanningReady, "prompt special token contract is incomplete"))
	if tensors != nil {
		ready := TensorReadinessFromInventory(*tensors)
		plan.TensorMetadataReady = ready.MetadataReady
	}
	add("tensor_inventory", plan.TensorMetadataReady, reasonIf(!plan.TensorMetadataReady, "missing text/vision/resampler tensor metadata"))
	if summary.AudioModelType != "" || summary.AudioHiddenSize > 0 || summary.AudioMelBins > 0 {
		add("audio_encoder_execution", false, "MiniCPM-O audio encoder tensor execution pending")
	}
	add("vision_tower_execution", false, "EVA02/SigLIP tensor execution pending")
	add("resampler_execution", false, "perceiver resampler tensor execution pending")
	add("embedding_injection", false, "image embedding injection into text embeddings pending")
	add("text_generation", false, "MiniCPM/Qwen2/Mistral text generation binding pending")
	plan.RuntimeReady = false
	return plan
}

func reasonIf(cond bool, reason string) string {
	if cond {
		return reason
	}
	return ""
}

func (p RuntimePlan) RequireReady() error {
	if p.RuntimeReady {
		return nil
	}
	var pending []string
	for _, op := range p.Ops {
		if !op.Ready {
			if op.Reason != "" {
				pending = append(pending, op.Name+": "+op.Reason)
			} else {
				pending = append(pending, op.Name)
			}
		}
	}
	if len(pending) == 0 {
		pending = append(pending, "runtime execution is not implemented")
	}
	return fmt.Errorf("MiniCPM-V/O runtime is not ready: %s", strings.Join(pending, "; "))
}

func processorPatch(p *config.MiniCPMVProcessorConfig) int {
	if p == nil {
		return 0
	}
	return p.PatchSize
}
