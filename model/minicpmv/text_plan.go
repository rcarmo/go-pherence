package minicpmv

import "github.com/rcarmo/go-pherence/loader/config"

type TextOp struct {
	Name   string `json:"name"`
	Ready  bool   `json:"ready"`
	Reason string `json:"reason,omitempty"`
}

type TextExecutionPlan struct {
	TextModelType string   `json:"text_model_type,omitempty"`
	HiddenSize    int      `json:"hidden_size"`
	Layers        int      `json:"layers"`
	Heads         int      `json:"heads"`
	KVHeads       int      `json:"kv_heads"`
	HeadDim       int      `json:"head_dim"`
	VocabSize     int      `json:"vocab_size"`
	Intermediate  int      `json:"intermediate_size"`
	HasEmbedding  bool     `json:"has_embedding"`
	HasLayers     bool     `json:"has_layers"`
	HasLMHead     bool     `json:"has_lm_head"`
	Generation    bool     `json:"generation_config"`
	MetadataReady bool     `json:"metadata_ready"`
	TensorReady   bool     `json:"tensor_ready"`
	Ready         bool     `json:"ready"`
	Ops           []TextOp `json:"ops"`
}

func BuildTextExecutionPlan(summary config.MiniCPMVSummary, generation *config.MiniCPMVGenerationConfig, tensors *TensorInventory) TextExecutionPlan {
	plan := TextExecutionPlan{TextModelType: summary.TextModelType, HiddenSize: summary.HiddenSize, Layers: summary.Layers, Heads: summary.Heads, KVHeads: summary.KVHeads, HeadDim: summary.HeadDim, VocabSize: summary.VocabSize, Intermediate: summary.IntermediateSize, Generation: generation != nil}
	plan.MetadataReady = summary.HiddenSize > 0 && summary.Layers > 0 && summary.Heads > 0 && summary.VocabSize > 0
	if tensors != nil {
		plan.HasEmbedding = tensors.Groups[TensorTextEmbedding] > 0
		plan.HasLayers = tensors.Groups[TensorTextLayer] > 0
		plan.HasLMHead = tensors.Groups[TensorTextLMHead] > 0
	}
	plan.TensorReady = plan.HasEmbedding && plan.HasLayers
	add := func(name string, ready bool, reason string) {
		plan.Ops = append(plan.Ops, TextOp{Name: name, Ready: ready, Reason: reasonIf(!ready, reason)})
	}
	add("text_metadata", plan.MetadataReady, "missing text dimensions")
	add("token_embedding", plan.HasEmbedding, "embedding tensor metadata missing")
	add("decoder_layers", plan.HasLayers, "decoder layer tensor metadata missing")
	add("lm_head", plan.HasLMHead, "LM head tensor metadata missing or tied embeddings not resolved")
	add("generation_config", plan.Generation, "generation_config.json missing; defaults required")
	add("prefill_decode", false, "MiniCPM/Qwen2/Mistral text prefill/decode binding pending")
	add("sampling", false, "generation sampling loop pending")
	plan.Ready = false
	return plan
}
