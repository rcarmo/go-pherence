package pockettts

import "fmt"

type fullTrainerTopology struct {
	flowLM                                                         *FlowLMTrainingCPU
	flow                                                           *FlowHeadCPU
	weighting                                                      *LSDWeightMLP
	transformerLayers, timeEmbeddings, flowBlocks, weightingLayers int
}

func (t *FullTrainer) refreshParameterBindings() error {
	if t.FlowLM != t.topology.flowLM || t.Flow != t.topology.flow || t.Weighting != t.topology.weighting || t.FlowLM == nil || t.FlowLM.Transformer == nil || len(t.FlowLM.Transformer.Layers) != t.topology.transformerLayers || len(t.Flow.Time) != t.topology.timeEmbeddings || len(t.Flow.Blocks) != t.topology.flowBlocks || len(t.Weighting.Layers) != t.topology.weightingLayers {
		return fmt.Errorf("Pocket TTS full trainer topology changed after construction")
	}
	for i := range t.bindings {
		binding := &t.bindings[i]
		live := fullLiveParameter(t, binding.name)
		if len(live) != len(binding.parameter) {
			return fmt.Errorf("Pocket TTS full parameter %q rebound with different shape", binding.name)
		}
		if len(live) > 0 && &live[0] != &binding.parameter[0] {
			if state, ok := t.Moments[binding.name]; ok && (len(state.M) != len(live) || len(state.V) != len(live)) {
				return fmt.Errorf("Pocket TTS full parameter %q rebound against incompatible state", binding.name)
			}
			binding.parameter = live
			t.params[binding.name] = live
			if t.EMADecay > 0 {
				shadow := t.EMA[binding.name]
				if len(shadow) != len(live) {
					return fmt.Errorf("Pocket TTS full parameter %q rebound against incompatible EMA", binding.name)
				}
			}
		}
	}
	return nil
}

func fullLiveParameter(t *FullTrainer, name string) []float32 {
	lm, flow, w := t.FlowLM, t.Flow, t.Weighting
	switch name {
	case "flow_lm.conditioner.embed.weight":
		return lm.Embedding
	case "flow_lm.bos_emb":
		return lm.BOS
	case "flow_lm.bos_before_voice":
		return lm.BOSBeforeVoice
	case "flow_lm.speaker_proj_weight":
		return lm.SpeakerProjection.Weight
	case "flow_lm.input_linear.weight":
		return lm.Input.Weight
	case "flow_lm.out_eos.weight":
		return lm.EOS.Weight
	case "flow_lm.out_eos.bias":
		return lm.EOS.Bias
	case "flow_lm.out_norm.weight":
		return lm.Transformer.FinalWeight
	case "flow_lm.out_norm.bias":
		return lm.Transformer.FinalBias
	case "flow_lm.flow_net.input_proj.weight":
		return flow.Input.Weight
	case "flow_lm.flow_net.input_proj.bias":
		return flow.Input.Bias
	case "flow_lm.flow_net.cond_embed.weight":
		return flow.Condition.Weight
	case "flow_lm.flow_net.cond_embed.bias":
		return flow.Condition.Bias
	case "flow_lm.flow_net.final_layer.linear.weight":
		return flow.Final.Linear.Weight
	case "flow_lm.flow_net.final_layer.linear.bias":
		return flow.Final.Linear.Bias
	case "flow_lm.flow_net.final_layer.adaLN_modulation.1.weight":
		return flow.Final.Modulation.Weight
	case "flow_lm.flow_net.final_layer.adaLN_modulation.1.bias":
		return flow.Final.Modulation.Bias
	}
	if index, suffix, ok := indexedSuffix(name, "flow_lm.transformer.layers."); ok && index < len(lm.Transformer.Layers) {
		l := lm.Transformer.Layers[index]
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
	if index, suffix, ok := indexedSuffix(name, "flow_lm.flow_net.time_embed."); ok && index < len(flow.Time) {
		l := flow.Time[index]
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
	if index, suffix, ok := indexedSuffix(name, "flow_lm.flow_net.res_blocks."); ok && index < len(flow.Blocks) {
		l := flow.Blocks[index]
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
	if index, suffix, ok := indexedSuffix(name, "flow.w_s_t."); ok && index%2 == 0 && index/2 < len(w.Layers) {
		l := w.Layers[index/2]
		switch suffix {
		case "weight":
			return l.Weight
		case "bias":
			return l.Bias
		}
	}
	return nil
}
