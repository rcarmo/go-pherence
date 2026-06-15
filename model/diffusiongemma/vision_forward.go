package diffusiongemma

import "strings"

type VisionGlobals struct {
	PatchInputProj         *TensorBinding `json:"patch_input_proj,omitempty"`
	PositionEmbeddingTable *TensorBinding `json:"position_embedding_table,omitempty"`
	StdBias                *TensorBinding `json:"std_bias,omitempty"`
	StdScale               *TensorBinding `json:"std_scale,omitempty"`
	EmbeddingProjection    *TensorBinding `json:"embedding_projection,omitempty"`
}

type VisionLayerBindings struct {
	Layer int    `json:"layer"`
	Type  string `json:"type,omitempty"`

	InputLayerNorm         *TensorBinding `json:"input_layernorm,omitempty"`
	PostAttentionLayerNorm *TensorBinding `json:"post_attention_layernorm,omitempty"`
	PreFFNLayerNorm        *TensorBinding `json:"pre_feedforward_layernorm,omitempty"`
	PostFFNLayerNorm       *TensorBinding `json:"post_feedforward_layernorm,omitempty"`

	QProj *TensorBinding `json:"q_proj,omitempty"`
	KProj *TensorBinding `json:"k_proj,omitempty"`
	VProj *TensorBinding `json:"v_proj,omitempty"`
	OProj *TensorBinding `json:"o_proj,omitempty"`
	QNorm *TensorBinding `json:"q_norm,omitempty"`
	KNorm *TensorBinding `json:"k_norm,omitempty"`

	MLPGateProj *TensorBinding `json:"mlp_gate_proj,omitempty"`
	MLPUpProj   *TensorBinding `json:"mlp_up_proj,omitempty"`
	MLPDownProj *TensorBinding `json:"mlp_down_proj,omitempty"`
}

type VisionForwardPlan struct {
	Globals VisionGlobals         `json:"globals"`
	Layers  []VisionLayerBindings `json:"layers"`
	Ready   bool                  `json:"ready"`
	Missing []string              `json:"missing,omitempty"`
}

func (w *VisionWeights) ForwardPlan() VisionForwardPlan {
	if w == nil {
		return VisionForwardPlan{Ready: false, Missing: []string{"nil vision weights"}}
	}
	plan := VisionForwardPlan{Ready: true}
	for i := range w.Globals {
		b := &w.Globals[i]
		switch {
		case strings.Contains(b.Name, "patch_embedder.input_proj"):
			plan.Globals.PatchInputProj = b
		case strings.Contains(b.Name, "position_embedding_table"):
			plan.Globals.PositionEmbeddingTable = b
		case strings.HasSuffix(b.Name, "std_bias"):
			plan.Globals.StdBias = b
		case strings.HasSuffix(b.Name, "std_scale"):
			plan.Globals.StdScale = b
		case strings.Contains(b.Name, "embed_vision.embedding_projection"):
			plan.Globals.EmbeddingProjection = b
		}
	}
	missingGlobal := func(name string, b *TensorBinding) {
		if b == nil {
			plan.Ready = false
			plan.Missing = append(plan.Missing, name)
		}
	}
	missingGlobal("patch_embedder.input_proj", plan.Globals.PatchInputProj)
	missingGlobal("patch_embedder.position_embedding_table", plan.Globals.PositionEmbeddingTable)
	missingGlobal("std_bias", plan.Globals.StdBias)
	missingGlobal("std_scale", plan.Globals.StdScale)
	missingGlobal("embed_vision.embedding_projection", plan.Globals.EmbeddingProjection)

	for _, layer := range w.Layers {
		lb := VisionLayerBindings{Layer: layer.Layer, Type: layer.Type}
		for i := range layer.Bindings {
			b := &layer.Bindings[i]
			assignVisionLayerBinding(&lb, b)
		}
		validateVisionLayerBinding(&plan, lb)
		plan.Layers = append(plan.Layers, lb)
	}
	return plan
}

func assignVisionLayerBinding(lb *VisionLayerBindings, b *TensorBinding) {
	name := b.Name
	switch {
	case strings.Contains(name, "input_layernorm"):
		lb.InputLayerNorm = b
	case strings.Contains(name, "post_attention_layernorm"):
		lb.PostAttentionLayerNorm = b
	case strings.Contains(name, "pre_feedforward_layernorm"):
		lb.PreFFNLayerNorm = b
	case strings.Contains(name, "post_feedforward_layernorm"):
		lb.PostFFNLayerNorm = b
	case strings.Contains(name, "self_attn.q_proj"):
		lb.QProj = b
	case strings.Contains(name, "self_attn.k_proj"):
		lb.KProj = b
	case strings.Contains(name, "self_attn.v_proj"):
		lb.VProj = b
	case strings.Contains(name, "self_attn.o_proj"):
		lb.OProj = b
	case strings.Contains(name, "self_attn.q_norm"):
		lb.QNorm = b
	case strings.Contains(name, "self_attn.k_norm"):
		lb.KNorm = b
	case strings.Contains(name, "mlp.gate_proj"):
		lb.MLPGateProj = b
	case strings.Contains(name, "mlp.up_proj"):
		lb.MLPUpProj = b
	case strings.Contains(name, "mlp.down_proj"):
		lb.MLPDownProj = b
	}
}

func validateVisionLayerBinding(plan *VisionForwardPlan, lb VisionLayerBindings) {
	require := func(role string, b *TensorBinding) {
		if b == nil {
			plan.Ready = false
			plan.Missing = append(plan.Missing, "vision layer "+itoa(lb.Layer)+" "+role)
		}
	}
	require("input_layernorm", lb.InputLayerNorm)
	require("post_attention_layernorm", lb.PostAttentionLayerNorm)
	require("pre_feedforward_layernorm", lb.PreFFNLayerNorm)
	require("post_feedforward_layernorm", lb.PostFFNLayerNorm)
	require("q_proj", lb.QProj)
	require("k_proj", lb.KProj)
	require("v_proj", lb.VProj)
	require("o_proj", lb.OProj)
	require("q_norm", lb.QNorm)
	require("k_norm", lb.KNorm)
	require("mlp.gate_proj", lb.MLPGateProj)
	require("mlp.up_proj", lb.MLPUpProj)
	require("mlp.down_proj", lb.MLPDownProj)
}
