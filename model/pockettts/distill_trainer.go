package pockettts

import (
	"fmt"
	"math"
	"sort"
)

// DistillTrainer applies AdamW/EMA only to the student conditioning/backbone
// subset upstream leaves requires_grad=true. EOS, flow head and w_s_t are not
// registered, so even decoupled weight decay cannot move them.
type DistillTrainer struct {
	Student     *FlowLMTrainingCPU
	Config      AdamWConfig
	EMADecay    float32
	StepCount   int
	Moments     map[string]NamedAdamState
	EMA         map[string][]float32
	names       []string
	params      map[string][]float32
	student     *FlowLMTrainingCPU
	transformer *TransformerCPU
	layers      int
}

func NewDistillTrainer(student *FlowLMTrainingCPU, config AdamWConfig, emaDecay float32) (*DistillTrainer, error) {
	if err := config.validate(); err != nil {
		return nil, err
	}
	if student == nil || student.Transformer == nil || !isFinite(emaDecay) || emaDecay < 0 || emaDecay >= 1 {
		return nil, fmt.Errorf("invalid Pocket TTS distill trainer")
	}
	params := distillParameterMap(student)
	for name, values := range params {
		if len(values) == 0 || !finiteF32(values) {
			return nil, fmt.Errorf("invalid Pocket TTS distill parameter %q", name)
		}
	}
	names := make([]string, 0, len(params))
	for name := range params {
		names = append(names, name)
	}
	sort.Strings(names)
	t := &DistillTrainer{Student: student, Config: config, EMADecay: emaDecay, Moments: map[string]NamedAdamState{}, names: names, params: params, student: student, transformer: student.Transformer, layers: len(student.Transformer.Layers)}
	if emaDecay > 0 {
		t.EMA = cloneTrainingMap(params)
	}
	return t, nil
}

func (t *DistillTrainer) Step(gradients *FlowLMTrainingGradients) error {
	if t == nil || gradients == nil {
		return fmt.Errorf("Pocket TTS distill trainer gradients are nil")
	}
	if t.Student == nil || t.Student != t.student || t.Student.Transformer == nil || t.Student.Transformer != t.transformer || len(t.Student.Transformer.Layers) != t.layers {
		return fmt.Errorf("Pocket TTS distill trainer topology changed")
	}
	params := distillParameterMap(t.Student)
	if len(params) != len(t.params) {
		return fmt.Errorf("Pocket TTS distill trainer topology changed")
	}
	for _, name := range t.names {
		if len(params[name]) != len(t.params[name]) {
			return fmt.Errorf("Pocket TTS distill parameter topology changed for %q", name)
		}
	}
	grads := distillGradientMap(gradients)
	for _, name := range t.names {
		parameter, ok := params[name]
		gradient, gok := grads[name]
		if !ok || !gok || len(parameter) != len(t.params[name]) || len(gradient) != len(parameter) || !finiteF32(gradient) {
			return fmt.Errorf("Pocket TTS distill gradient shape mismatch for %q", name)
		}
		if state, exists := t.Moments[name]; exists && (len(state.M) != len(parameter) || len(state.V) != len(parameter)) {
			return fmt.Errorf("Pocket TTS distill Adam shape mismatch for %q", name)
		}
		if t.EMADecay > 0 && len(t.EMA[name]) != len(parameter) {
			return fmt.Errorf("Pocket TTS distill EMA shape mismatch for %q", name)
		}
	}
	grads = cloneTrainingMap(grads)
	step := t.StepCount + 1
	b1 := float32(1 - math.Pow(float64(t.Config.Beta1), float64(step)))
	b2 := float32(1 - math.Pow(float64(t.Config.Beta2), float64(step)))
	for _, name := range t.names {
		parameter, gradient := params[name], grads[name]
		state, ok := t.Moments[name]
		if !ok {
			state = NamedAdamState{Name: name, M: make([]float32, len(parameter)), V: make([]float32, len(parameter))}
		}
		for i := range parameter {
			g := gradient[i]
			state.M[i] = t.Config.Beta1*state.M[i] + (1-t.Config.Beta1)*g
			state.V[i] = t.Config.Beta2*state.V[i] + (1-t.Config.Beta2)*g*g
			parameter[i] *= 1 - t.Config.LearningRate*t.Config.WeightDecay
			parameter[i] -= t.Config.LearningRate * (state.M[i] / b1) / (float32(math.Sqrt(float64(state.V[i]/b2))) + t.Config.Epsilon)
		}
		t.Moments[name] = state
		if t.EMADecay > 0 {
			shadow := t.EMA[name]
			for i := range parameter {
				shadow[i] = t.EMADecay*shadow[i] + (1-t.EMADecay)*parameter[i]
			}
		}
	}
	t.StepCount, t.params = step, params
	return nil
}

func distillParameterMap(m *FlowLMTrainingCPU) map[string][]float32 {
	out := map[string][]float32{"flow_lm.conditioner.embed.weight": m.Embedding, "flow_lm.bos_emb": m.BOS, "flow_lm.bos_before_voice": m.BOSBeforeVoice, "flow_lm.speaker_proj_weight": m.SpeakerProjection.Weight, "flow_lm.input_linear.weight": m.Input.Weight}
	addTransformerParameters(out, m.Transformer)
	return out
}
func distillGradientMap(g *FlowLMTrainingGradients) map[string][]float32 {
	out := map[string][]float32{"flow_lm.conditioner.embed.weight": g.Embedding, "flow_lm.bos_emb": g.BOS, "flow_lm.bos_before_voice": g.BOSBeforeVoice, "flow_lm.speaker_proj_weight": g.SpeakerProjection.Weight, "flow_lm.input_linear.weight": g.Input.Weight}
	addDistillTransformerGradients(out, g.Transformer)
	return out
}
func addDistillTransformerGradients(out map[string][]float32, g *TransformerGradients) {
	if g == nil {
		return
	}
	for i, l := range g.Layers {
		p := fmt.Sprintf("flow_lm.transformer.layers.%d", i)
		out[p+".norm1.weight"], out[p+".norm1.bias"] = l.Norm1Weight, l.Norm1Bias
		out[p+".norm2.weight"], out[p+".norm2.bias"] = l.Norm2Weight, l.Norm2Bias
		out[p+".self_attn.in_proj.weight"], out[p+".self_attn.out_proj.weight"] = l.InProjection.Weight, l.OutProjection.Weight
		out[p+".linear1.weight"], out[p+".linear2.weight"] = l.FC1.Weight, l.FC2.Weight
		if l.LayerScale1 != nil {
			out[p+".layer_scale_1.scale"] = l.LayerScale1
		}
		if l.LayerScale2 != nil {
			out[p+".layer_scale_2.scale"] = l.LayerScale2
		}
	}
	out["flow_lm.out_norm.weight"], out["flow_lm.out_norm.bias"] = g.FinalWeight, g.FinalBias
}
