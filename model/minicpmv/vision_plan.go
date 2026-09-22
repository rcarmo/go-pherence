package minicpmv

import (
	"strings"

	"github.com/rcarmo/go-pherence/loader/config"
)

type VisionOp struct {
	Name   string `json:"name"`
	Ready  bool   `json:"ready"`
	Reason string `json:"reason,omitempty"`
}

type VisionExecutionPlan struct {
	VisionModelType string     `json:"vision_model_type"`
	ImageSize       int        `json:"image_size"`
	PatchSize       int        `json:"patch_size"`
	PatchGrid       int        `json:"patch_grid"`
	VisionTokens    int        `json:"vision_tokens"`
	VisionHidden    int        `json:"vision_hidden"`
	VisionLayers    int        `json:"vision_layers"`
	ResamplerQuery  int        `json:"resampler_query"`
	ResamplerGrid   int        `json:"resampler_grid"`
	ResamplerHeads  int        `json:"resampler_heads"`
	TextHidden      int        `json:"text_hidden"`
	NeedsKVProj     bool       `json:"needs_kv_proj"`
	Ready           bool       `json:"ready"`
	Ops             []VisionOp `json:"ops"`
}

func BuildVisionExecutionPlan(summary config.MiniCPMVSummary, tensors *TensorInventory) VisionExecutionPlan {
	plan := VisionExecutionPlan{
		VisionModelType: summary.VisionModelType,
		ImageSize:       summary.ImageSize,
		PatchSize:       summary.PatchSize,
		VisionHidden:    summary.VisionHiddenSize,
		VisionLayers:    summary.VisionLayers,
		ResamplerQuery:  summary.NumQuery,
		ResamplerGrid:   summary.ResamplerGrid,
		ResamplerHeads:  summary.ResamplerHeads,
		TextHidden:      summary.HiddenSize,
		NeedsKVProj:     summary.VisionHiddenSize > 0 && summary.HiddenSize > 0 && summary.VisionHiddenSize != summary.HiddenSize,
	}
	if summary.ImageSize > 0 && summary.PatchSize > 0 && summary.ImageSize%summary.PatchSize == 0 {
		plan.PatchGrid = summary.ImageSize / summary.PatchSize
		plan.VisionTokens = plan.PatchGrid * plan.PatchGrid
	}
	hasVision, hasResampler, hasProjector := false, false, false
	if tensors != nil {
		hasVision = tensors.Groups[TensorVisionTower] > 0
		hasResampler = tensors.Groups[TensorResampler] > 0
		hasProjector = tensors.Groups[TensorProjector] > 0
	}
	add := func(name string, ready bool, reason string) {
		plan.Ops = append(plan.Ops, VisionOp{Name: name, Ready: ready, Reason: reasonIf(!ready, reason)})
	}
	add("image_preprocess_bchw", plan.ImageSize > 0 && plan.PatchSize > 0, "missing image/patch size")
	add("patch_embedding", hasVision && plan.PatchGrid > 0, "vision patch embedding tensors or patch grid missing")
	add("siglip_cpu_reference", IsLikelySigLIPVision(summary) && hasVision && summary.VisionLayers > 0 && summary.VisionHiddenSize > 0, "SigLIP tensor metadata or dimensions missing")
	add("eva_cpu_reference", false, "EVA02 tensor execution pending")
	add("vision_transformer", false, "SigLIP synthetic execution implemented; EVA02 and released-model parity pending")
	add("vision_token_select", true, "")
	add("resampler_queries", hasResampler && summary.NumQuery > 0, "resampler query tensors or num_query missing")
	add("resampler_cross_attention", hasResampler && summary.ResamplerHeads > 0, "resampler attention tensors or heads missing")
	add("resampler_kv_projection", !plan.NeedsKVProj || hasResampler || hasProjector, "vision/text hidden sizes differ and kv/projector tensors are missing")
	add("resampler_cpu_reference", hasResampler && summary.NumQuery > 0 && summary.ResamplerHeads > 0 && (!plan.NeedsKVProj || hasResampler || hasProjector), "resampler metadata or tensors missing")
	add("language_embedding_injection", true, "")
	plan.Ready = false
	return plan
}

func IsLikelySigLIPVision(summary config.MiniCPMVSummary) bool {
	return strings.Contains(strings.ToLower(summary.VisionModelType), "siglip")
}

func IsLikelyEVAVision(summary config.MiniCPMVSummary) bool {
	v := strings.ToLower(summary.VisionModelType + " " + summary.Architecture)
	return strings.Contains(v, "eva") || strings.Contains(v, "clip") && summary.VisionModelType == ""
}
