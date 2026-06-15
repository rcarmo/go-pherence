package diffusiongemma

import "strings"

// TextGlobals binds non-layer decoder tensors by semantic role.
type TextGlobals struct {
	EmbedTokens      *TensorBinding `json:"embed_tokens,omitempty"`
	FinalNorm        *TensorBinding `json:"final_norm,omitempty"`
	RopeFreqs        *TensorBinding `json:"rope_freqs,omitempty"`
	SelfCondPreNorm  *TensorBinding `json:"self_conditioning_pre_norm,omitempty"`
	SelfCondGateProj *TensorBinding `json:"self_conditioning_gate_proj,omitempty"`
	SelfCondUpProj   *TensorBinding `json:"self_conditioning_up_proj,omitempty"`
	SelfCondDownProj *TensorBinding `json:"self_conditioning_down_proj,omitempty"`
}

// TextLayerBindings exposes semantic handles for one decoder layer. It is a
// metadata scaffold for future native forward execution; nil fields mean the
// role is absent for that layer type.
type TextLayerBindings struct {
	Layer int    `json:"layer"`
	Type  string `json:"type,omitempty"`

	InputLayerNorm         *TensorBinding `json:"input_layernorm,omitempty"`
	PostAttentionLayerNorm *TensorBinding `json:"post_attention_layernorm,omitempty"`
	PreFFNLayerNorm        *TensorBinding `json:"pre_feedforward_layernorm,omitempty"`
	PreFFNLayerNorm2       *TensorBinding `json:"pre_feedforward_layernorm_2,omitempty"`
	PostFFNLayerNorm       *TensorBinding `json:"post_feedforward_layernorm,omitempty"`
	PostFFNLayerNorm1      *TensorBinding `json:"post_feedforward_layernorm_1,omitempty"`
	PostFFNLayerNorm2      *TensorBinding `json:"post_feedforward_layernorm_2,omitempty"`
	LayerScalar            *TensorBinding `json:"layer_scalar,omitempty"`
	EncLayerScalar         *TensorBinding `json:"enc_layer_scalar,omitempty"`

	QProj *TensorBinding `json:"q_proj,omitempty"`
	KProj *TensorBinding `json:"k_proj,omitempty"`
	VProj *TensorBinding `json:"v_proj,omitempty"`
	OProj *TensorBinding `json:"o_proj,omitempty"`
	QNorm *TensorBinding `json:"q_norm,omitempty"`
	KNorm *TensorBinding `json:"k_norm,omitempty"`

	MLPGateProj *TensorBinding `json:"mlp_gate_proj,omitempty"`
	MLPUpProj   *TensorBinding `json:"mlp_up_proj,omitempty"`
	MLPDownProj *TensorBinding `json:"mlp_down_proj,omitempty"`

	RouterProj           *TensorBinding `json:"router_proj,omitempty"`
	RouterScale          *TensorBinding `json:"router_scale,omitempty"`
	RouterPerExpertScale *TensorBinding `json:"router_per_expert_scale,omitempty"`
	ExpertsGateUpProj    *TensorBinding `json:"experts_gate_up_proj,omitempty"`
	ExpertsDownProj      *TensorBinding `json:"experts_down_proj,omitempty"`
}

// TextForwardPlan is a semantic view over TextWeights for future forward code.
type TextForwardPlan struct {
	Globals        TextGlobals         `json:"globals"`
	Layers         []TextLayerBindings `json:"layers"`
	IndexedExperts bool                `json:"indexed_experts,omitempty"`
	Ready          bool                `json:"ready"`
	Missing        []string            `json:"missing,omitempty"`
}

func (w *TextWeights) ForwardPlan() TextForwardPlan {
	plan := TextForwardPlan{Ready: true, IndexedExperts: w.IndexedExperts}
	if w == nil {
		return TextForwardPlan{Ready: false, Missing: []string{"nil text weights"}}
	}
	for i := range w.Globals {
		b := &w.Globals[i]
		switch {
		case strings.HasSuffix(b.Name, "embed_tokens.weight"):
			plan.Globals.EmbedTokens = b
		case strings.HasSuffix(b.Name, "decoder.norm.weight"):
			plan.Globals.FinalNorm = b
		case strings.Contains(b.Name, "rope_freqs"):
			plan.Globals.RopeFreqs = b
		case strings.Contains(b.Name, "self_conditioning.pre_norm"):
			plan.Globals.SelfCondPreNorm = b
		case strings.Contains(b.Name, "self_conditioning.gate_proj"):
			plan.Globals.SelfCondGateProj = b
		case strings.Contains(b.Name, "self_conditioning.up_proj"):
			plan.Globals.SelfCondUpProj = b
		case strings.Contains(b.Name, "self_conditioning.down_proj"):
			plan.Globals.SelfCondDownProj = b
		}
	}
	missingGlobal := func(name string, b *TensorBinding) {
		if b == nil {
			plan.Ready = false
			plan.Missing = append(plan.Missing, name)
		}
	}
	missingGlobal("embed_tokens", plan.Globals.EmbedTokens)
	missingGlobal("final_norm", plan.Globals.FinalNorm)
	missingGlobal("self_conditioning.pre_norm", plan.Globals.SelfCondPreNorm)
	missingGlobal("self_conditioning.gate_proj", plan.Globals.SelfCondGateProj)
	missingGlobal("self_conditioning.up_proj", plan.Globals.SelfCondUpProj)
	missingGlobal("self_conditioning.down_proj", plan.Globals.SelfCondDownProj)

	for _, layer := range w.Layers {
		lb := TextLayerBindings{Layer: layer.Layer, Type: layer.Type}
		for i := range layer.Bindings {
			b := &layer.Bindings[i]
			assignLayerBinding(&lb, b)
		}
		validateLayerBinding(&plan, lb)
		plan.Layers = append(plan.Layers, lb)
	}
	return plan
}

func assignLayerBinding(lb *TextLayerBindings, b *TensorBinding) {
	name := b.Name
	switch {
	case strings.Contains(name, "input_layernorm"):
		lb.InputLayerNorm = b
	case strings.Contains(name, "post_attention_layernorm"):
		lb.PostAttentionLayerNorm = b
	case strings.Contains(name, "pre_feedforward_layernorm_2"):
		lb.PreFFNLayerNorm2 = b
	case strings.Contains(name, "pre_feedforward_layernorm.weight"):
		lb.PreFFNLayerNorm = b
	case strings.Contains(name, "post_feedforward_layernorm_1"):
		lb.PostFFNLayerNorm1 = b
	case strings.Contains(name, "post_feedforward_layernorm_2"):
		lb.PostFFNLayerNorm2 = b
	case strings.Contains(name, "post_feedforward_layernorm.weight"):
		lb.PostFFNLayerNorm = b
	case strings.Contains(name, "model.encoder.language_model.layers.") && strings.HasSuffix(name, ".layer_scalar"):
		lb.EncLayerScalar = b
	case strings.Contains(name, "enc_layer_output_scale") || strings.Contains(name, "enc_layer_scalar"):
		lb.EncLayerScalar = b
	case strings.HasSuffix(name, ".layer_scalar"):
		lb.LayerScalar = b
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
	case strings.Contains(name, "router.proj"):
		lb.RouterProj = b
	case strings.Contains(name, "router.per_expert_scale"):
		lb.RouterPerExpertScale = b
	case strings.Contains(name, "router.scale"):
		lb.RouterScale = b
	case strings.Contains(name, "experts.gate_up_proj"):
		lb.ExpertsGateUpProj = b
	case strings.Contains(name, "experts.down_proj"):
		lb.ExpertsDownProj = b
	}
}

func validateLayerBinding(plan *TextForwardPlan, lb TextLayerBindings) {
	require := func(role string, b *TensorBinding) {
		if b == nil {
			plan.Ready = false
			plan.Missing = append(plan.Missing, "layer "+itoa(lb.Layer)+" "+role)
		}
	}
	require("input_layernorm", lb.InputLayerNorm)
	require("post_attention_layernorm", lb.PostAttentionLayerNorm)
	require("pre_feedforward_layernorm", lb.PreFFNLayerNorm)
	require("pre_feedforward_layernorm_2", lb.PreFFNLayerNorm2)
	require("post_feedforward_layernorm", lb.PostFFNLayerNorm)
	require("post_feedforward_layernorm_1", lb.PostFFNLayerNorm1)
	require("post_feedforward_layernorm_2", lb.PostFFNLayerNorm2)
	require("layer_scalar", lb.LayerScalar)
	require("q_proj", lb.QProj)
	require("k_proj", lb.KProj)
	if lb.Type != "full_attention" {
		require("v_proj", lb.VProj)
	}
	require("o_proj", lb.OProj)
	require("q_norm", lb.QNorm)
	require("k_norm", lb.KNorm)
	require("mlp_gate_proj", lb.MLPGateProj)
	require("mlp_up_proj", lb.MLPUpProj)
	require("mlp_down_proj", lb.MLPDownProj)
	require("router_proj", lb.RouterProj)
	require("router_scale", lb.RouterScale)
	require("router_per_expert_scale", lb.RouterPerExpertScale)
	if !plan.IndexedExperts {
		require("experts_gate_up_proj", lb.ExpertsGateUpProj)
		require("experts_down_proj", lb.ExpertsDownProj)
	}
}
