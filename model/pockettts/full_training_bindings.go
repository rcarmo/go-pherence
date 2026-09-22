package pockettts

import "strings"

type fullParamBinding struct {
	name      string
	parameter []float32
}

func makeFullParamBindings(names []string, params map[string][]float32) []fullParamBinding {
	out := make([]fullParamBinding, len(names))
	for i, name := range names {
		out[i] = fullParamBinding{name: name, parameter: params[name]}
	}
	return out
}

func (b fullParamBinding) gradient(g TrainingStepGradients) []float32 {
	return fullGradientByName(g, b.name)
}

func fullGradientByName(g TrainingStepGradients, name string) []float32 {
	switch name {
	case "flow_lm.conditioner.embed.weight":
		return g.FlowLM.Embedding
	case "flow_lm.bos_emb":
		return g.FlowLM.BOS
	case "flow_lm.bos_before_voice":
		return g.FlowLM.BOSBeforeVoice
	case "flow_lm.speaker_proj_weight":
		return g.FlowLM.SpeakerProjection.Weight
	case "flow_lm.input_linear.weight":
		return g.FlowLM.Input.Weight
	case "flow_lm.out_eos.weight":
		return g.FlowLM.EOS.Weight
	case "flow_lm.out_eos.bias":
		return g.FlowLM.EOS.Bias
	case "flow_lm.out_norm.weight":
		return g.FlowLM.Transformer.FinalWeight
	case "flow_lm.out_norm.bias":
		return g.FlowLM.Transformer.FinalBias
	case "flow_lm.flow_net.input_proj.weight":
		return g.Flow.Input.Weight
	case "flow_lm.flow_net.input_proj.bias":
		return g.Flow.Input.Bias
	case "flow_lm.flow_net.cond_embed.weight":
		return g.Flow.Condition.Weight
	case "flow_lm.flow_net.cond_embed.bias":
		return g.Flow.Condition.Bias
	case "flow_lm.flow_net.final_layer.linear.weight":
		return g.Flow.Final.Linear.Weight
	case "flow_lm.flow_net.final_layer.linear.bias":
		return g.Flow.Final.Linear.Bias
	case "flow_lm.flow_net.final_layer.adaLN_modulation.1.weight":
		return g.Flow.Final.Modulation.Weight
	case "flow_lm.flow_net.final_layer.adaLN_modulation.1.bias":
		return g.Flow.Final.Modulation.Bias
	}
	if index, suffix, ok := indexedSuffix(name, "flow_lm.transformer.layers."); ok && index < len(g.FlowLM.Transformer.Layers) {
		l := g.FlowLM.Transformer.Layers[index]
		switch suffix {
		case "norm1.weight":
			return l.Norm1Weight
		case "norm1.bias":
			return l.Norm1Bias
		case "norm2.weight":
			return l.Norm2Weight
		case "norm2.bias":
			return l.Norm2Bias
		case "self_attn.in_proj.weight":
			return l.InProjection.Weight
		case "self_attn.out_proj.weight":
			return l.OutProjection.Weight
		case "linear1.weight":
			return l.FC1.Weight
		case "linear2.weight":
			return l.FC2.Weight
		case "layer_scale_1.scale":
			return l.LayerScale1
		case "layer_scale_2.scale":
			return l.LayerScale2
		}
	}
	if index, suffix, ok := indexedSuffix(name, "flow_lm.flow_net.time_embed."); ok && index < len(g.Flow.Time) {
		l := g.Flow.Time[index]
		switch suffix {
		case "mlp.0.weight":
			return l.FC1.Weight
		case "mlp.0.bias":
			return l.FC1.Bias
		case "mlp.2.weight":
			return l.FC2.Weight
		case "mlp.2.bias":
			return l.FC2.Bias
		case "mlp.3.alpha":
			return l.RMSWeight
		}
	}
	if index, suffix, ok := indexedSuffix(name, "flow_lm.flow_net.res_blocks."); ok && index < len(g.Flow.Blocks) {
		l := g.Flow.Blocks[index]
		switch suffix {
		case "in_ln.weight":
			return l.NormWeight
		case "in_ln.bias":
			return l.NormBias
		case "mlp.0.weight":
			return l.FC1.Weight
		case "mlp.0.bias":
			return l.FC1.Bias
		case "mlp.2.weight":
			return l.FC2.Weight
		case "mlp.2.bias":
			return l.FC2.Bias
		case "adaLN_modulation.1.weight":
			return l.Modulation.Weight
		case "adaLN_modulation.1.bias":
			return l.Modulation.Bias
		}
	}
	if index, suffix, ok := indexedSuffix(name, "flow.w_s_t."); ok && index%2 == 0 && index/2 < len(g.Weighting.Layers) {
		l := g.Weighting.Layers[index/2]
		switch suffix {
		case "weight":
			return l.Weight
		case "bias":
			return l.Bias
		}
	}
	return nil
}

func indexedSuffix(name, prefix string) (int, string, bool) {
	if !strings.HasPrefix(name, prefix) {
		return 0, "", false
	}
	rest := name[len(prefix):]
	dot := strings.IndexByte(rest, '.')
	if dot <= 0 {
		return 0, "", false
	}
	index := 0
	for _, c := range []byte(rest[:dot]) {
		if c < '0' || c > '9' {
			return 0, "", false
		}
		index = index*10 + int(c-'0')
	}
	return index, rest[dot+1:], true
}
