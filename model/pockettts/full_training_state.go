package pockettts

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
)

// FullTrainingState is the resumable named-tensor state for the native FlowLM,
// flow head and LSD weighting network.
type FullTrainingState struct {
	Version  int                   `json:"version"`
	Step     int                   `json:"step"`
	AdamW    AdamWConfig           `json:"adamw"`
	EMADecay float32               `json:"ema_decay,omitempty"`
	Params   []NamedTrainingTensor `json:"params"`
	Buffers  []NamedTrainingTensor `json:"buffers"`
	Adam     []NamedAdamState      `json:"adam"`
	EMA      []NamedTrainingTensor `json:"ema,omitempty"`
}

type FullTrainer struct {
	FlowLM    *FlowLMTrainingCPU
	Flow      *FlowHeadCPU
	Weighting *LSDWeightMLP
	Config    AdamWConfig
	EMADecay  float32
	StepCount int
	Moments   map[string]NamedAdamState
	EMA       map[string][]float32
}

func NewFullTrainer(flowLM *FlowLMTrainingCPU, flow *FlowHeadCPU, weighting *LSDWeightMLP, config AdamWConfig, emaDecay float32) (*FullTrainer, error) {
	if err := config.validate(); err != nil {
		return nil, err
	}
	if !isFinite(emaDecay) || emaDecay < 0 || emaDecay >= 1 {
		return nil, fmt.Errorf("Pocket TTS full EMA decay must be in [0,1)")
	}
	params, err := fullParameterMap(flowLM, flow, weighting)
	if err != nil {
		return nil, err
	}
	t := &FullTrainer{FlowLM: flowLM, Flow: flow, Weighting: weighting, Config: config, EMADecay: emaDecay, Moments: map[string]NamedAdamState{}}
	if emaDecay > 0 {
		t.EMA = cloneTrainingMap(params)
	}
	return t, nil
}

func (t *FullTrainer) Step(gradients TrainingStepGradients) error {
	if t == nil {
		return fmt.Errorf("Pocket TTS full trainer is nil")
	}
	params, err := fullParameterMap(t.FlowLM, t.Flow, t.Weighting)
	if err != nil {
		return err
	}
	grads, err := fullGradientMap(gradients)
	if err != nil {
		return err
	}
	if len(params) != len(grads) {
		return fmt.Errorf("Pocket TTS full gradient count=%d want=%d", len(grads), len(params))
	}
	for name, parameter := range params {
		gradient, ok := grads[name]
		if !ok || len(gradient) != len(parameter) {
			return fmt.Errorf("Pocket TTS full gradient shape mismatch for %q", name)
		}
		for _, value := range gradient {
			if !isFinite(value) {
				return fmt.Errorf("Pocket TTS full gradient %q is non-finite", name)
			}
		}
		if state, ok := t.Moments[name]; ok && (len(state.M) != len(parameter) || len(state.V) != len(parameter)) {
			return fmt.Errorf("Pocket TTS full AdamW shape mismatch for %q", name)
		}
		if t.EMADecay > 0 && t.EMA != nil && len(t.EMA[name]) != len(parameter) {
			return fmt.Errorf("Pocket TTS full EMA shape mismatch for %q", name)
		}
	}
	step := t.StepCount + 1
	beta1Correction := float32(1 - math.Pow(float64(t.Config.Beta1), float64(step)))
	beta2Correction := float32(1 - math.Pow(float64(t.Config.Beta2), float64(step)))
	for _, name := range sortedTrainingKeys(params) {
		parameter, gradient := params[name], grads[name]
		state, ok := t.Moments[name]
		if !ok {
			state = NamedAdamState{Name: name, M: make([]float32, len(parameter)), V: make([]float32, len(parameter))}
		}
		for i := range parameter {
			g := gradient[i]
			state.M[i] = t.Config.Beta1*state.M[i] + (1-t.Config.Beta1)*g
			state.V[i] = t.Config.Beta2*state.V[i] + (1-t.Config.Beta2)*g*g
			mHat := state.M[i] / beta1Correction
			vHat := state.V[i] / beta2Correction
			parameter[i] *= 1 - t.Config.LearningRate*t.Config.WeightDecay
			parameter[i] -= t.Config.LearningRate * mHat / (float32(math.Sqrt(float64(vHat))) + t.Config.Epsilon)
		}
		t.Moments[name] = state
	}
	t.StepCount = step
	if t.EMADecay > 0 {
		if t.EMA == nil {
			t.EMA = cloneTrainingMap(params)
		} else {
			for name, parameter := range params {
				shadow := t.EMA[name]
				for i := range parameter {
					shadow[i] = t.EMADecay*shadow[i] + (1-t.EMADecay)*parameter[i]
				}
			}
		}
	}
	return nil
}

func (t *FullTrainer) State() (FullTrainingState, error) {
	if t == nil {
		return FullTrainingState{}, fmt.Errorf("Pocket TTS full trainer is nil")
	}
	params, err := fullParameterMap(t.FlowLM, t.Flow, t.Weighting)
	if err != nil {
		return FullTrainingState{}, err
	}
	s := FullTrainingState{Version: 1, Step: t.StepCount, AdamW: t.Config, EMADecay: t.EMADecay, Params: trainingMapToList(params), Buffers: trainingMapToList(fullBufferMap(t.FlowLM, t.Flow))}
	for _, name := range sortedAdamKeys(t.Moments) {
		v := t.Moments[name]
		s.Adam = append(s.Adam, NamedAdamState{Name: name, M: append([]float32(nil), v.M...), V: append([]float32(nil), v.V...)})
	}
	if t.EMADecay > 0 {
		s.EMA = trainingMapToList(t.EMA)
	}
	return s, s.Validate()
}
func (s FullTrainingState) Validate() error {
	if s.Version != 1 || s.Step < 0 || !isFinite(s.EMADecay) || s.EMADecay < 0 || s.EMADecay >= 1 {
		return fmt.Errorf("invalid Pocket TTS full training state header")
	}
	if err := s.AdamW.validate(); err != nil {
		return err
	}
	specs := map[string][]float32{}
	for _, p := range s.Params {
		if p.Name == "" || specs[p.Name] != nil || len(p.Values) == 0 {
			return fmt.Errorf("invalid Pocket TTS full parameter %q", p.Name)
		}
		for _, v := range p.Values {
			if !isFinite(v) {
				return fmt.Errorf("Pocket TTS full parameter %q is non-finite", p.Name)
			}
		}
		specs[p.Name] = make([]float32, len(p.Values))
	}
	if len(specs) == 0 {
		return fmt.Errorf("Pocket TTS full training state has no parameters")
	}
	buffers := map[string][]float32{}
	for _, buffer := range s.Buffers {
		if buffer.Name == "" || buffers[buffer.Name] != nil || len(buffer.Values) == 0 {
			return fmt.Errorf("invalid Pocket TTS full buffer %q", buffer.Name)
		}
		for _, value := range buffer.Values {
			if !isFinite(value) {
				return fmt.Errorf("Pocket TTS full buffer %q is non-finite", buffer.Name)
			}
		}
		buffers[buffer.Name] = buffer.Values
	}
	if len(buffers) == 0 {
		return fmt.Errorf("Pocket TTS full training state has no buffers")
	}
	if _, err := adamListToMap(s.Adam, specs, s.Step > 0); err != nil {
		return err
	}
	if s.EMADecay == 0 {
		if len(s.EMA) != 0 {
			return fmt.Errorf("Pocket TTS full state has EMA with zero decay")
		}
	} else if _, err := tensorListToMap(s.EMA, specs, "full EMA"); err != nil {
		return err
	}
	return nil
}

func (t *FullTrainer) LoadState(s FullTrainingState) error {
	if t == nil {
		return fmt.Errorf("Pocket TTS full trainer is nil")
	}
	if err := s.Validate(); err != nil {
		return err
	}
	params, err := fullParameterMap(t.FlowLM, t.Flow, t.Weighting)
	if err != nil {
		return err
	}
	loaded, err := tensorListToMap(s.Params, params, "full parameter")
	if err != nil {
		return err
	}
	buffers := fullBufferMap(t.FlowLM, t.Flow)
	loadedBuffers, err := tensorListToMap(s.Buffers, buffers, "full buffer")
	if err != nil {
		return err
	}
	for name, p := range params {
		copy(p, loaded[name])
	}
	for name, buffer := range buffers {
		copy(buffer, loadedBuffers[name])
	}
	t.Config = s.AdamW
	t.EMADecay = s.EMADecay
	t.StepCount = s.Step
	t.Moments, _ = adamListToMap(s.Adam, params, s.Step > 0)
	if s.EMADecay > 0 {
		t.EMA, _ = tensorListToMap(s.EMA, params, "full EMA")
	} else {
		t.EMA = nil
	}
	return nil
}

func SaveFullTrainingState(path string, s FullTrainingState) error {
	if err := s.Validate(); err != nil {
		return err
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	f, err := os.CreateTemp(dir, ".pockettts-full-*.tmp")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	if err = json.NewEncoder(f).Encode(s); err != nil {
		f.Close()
		return err
	}
	if err = f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	if err = os.Rename(f.Name(), path); err != nil {
		return err
	}
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	if err = d.Sync(); err != nil {
		d.Close()
		return err
	}
	return d.Close()
}
func LoadFullTrainingState(path string) (FullTrainingState, error) {
	f, err := os.Open(path)
	if err != nil {
		return FullTrainingState{}, err
	}
	defer f.Close()
	return ReadFullTrainingState(f)
}
func ReadFullTrainingState(r io.Reader) (FullTrainingState, error) {
	var s FullTrainingState
	if r == nil {
		return s, fmt.Errorf("Pocket TTS full state reader is nil")
	}
	d := json.NewDecoder(io.LimitReader(r, 512<<20))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		return s, err
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return s, fmt.Errorf("Pocket TTS full state contains trailing data")
	}
	return s, s.Validate()
}

func fullParameterMap(lm *FlowLMTrainingCPU, flow *FlowHeadCPU, w *LSDWeightMLP) (map[string][]float32, error) {
	if lm == nil || flow == nil || w == nil {
		return nil, fmt.Errorf("Pocket TTS full model is nil")
	}
	out := map[string][]float32{}
	addFlowLMParameters(out, lm)
	addFlowHeadParameters(out, flow)
	for i, l := range w.Layers {
		p := fmt.Sprintf("flow.w_s_t.%d", 2*i)
		out[p+".weight"], out[p+".bias"] = l.Weight, l.Bias
	}
	for name, v := range out {
		if len(v) == 0 {
			return nil, fmt.Errorf("Pocket TTS full parameter %q is empty", name)
		}
	}
	return out, nil
}
func fullGradientMap(g TrainingStepGradients) (map[string][]float32, error) {
	if g.FlowLM == nil || g.Flow == nil || g.Weighting == nil {
		return nil, fmt.Errorf("Pocket TTS full gradients are incomplete")
	}
	out := map[string][]float32{}
	addFlowLMGradients(out, g.FlowLM)
	addFlowHeadGradientParameters(out, g.Flow)
	for i, l := range g.Weighting.Layers {
		p := fmt.Sprintf("flow.w_s_t.%d", 2*i)
		out[p+".weight"], out[p+".bias"] = l.Weight, l.Bias
	}
	return out, nil
}
func fullBufferMap(lm *FlowLMTrainingCPU, flow *FlowHeadCPU) map[string][]float32 {
	out := map[string][]float32{"flow_lm.emb_mean": lm.LatentMean, "flow_lm.emb_std": lm.LatentStd}
	for i, timestep := range flow.Time {
		out[fmt.Sprintf("flow_lm.flow_net.time_embed.%d.freqs", i)] = timestep.Frequencies
	}
	return out
}

func addFlowLMParameters(out map[string][]float32, m *FlowLMTrainingCPU) {
	out["flow_lm.conditioner.embed.weight"] = m.Embedding
	out["flow_lm.bos_emb"] = m.BOS
	out["flow_lm.bos_before_voice"] = m.BOSBeforeVoice
	out["flow_lm.speaker_proj_weight"] = m.SpeakerProjection.Weight
	out["flow_lm.input_linear.weight"] = m.Input.Weight
	out["flow_lm.out_eos.weight"], out["flow_lm.out_eos.bias"] = m.EOS.Weight, m.EOS.Bias
	addTransformerParameters(out, m.Transformer)
}
func addTransformerParameters(out map[string][]float32, m *TransformerCPU) {
	for i, l := range m.Layers {
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
	out["flow_lm.out_norm.weight"], out["flow_lm.out_norm.bias"] = m.FinalWeight, m.FinalBias
}
func addFlowHeadParameters(out map[string][]float32, m *FlowHeadCPU) {
	out["flow_lm.flow_net.input_proj.weight"], out["flow_lm.flow_net.input_proj.bias"] = m.Input.Weight, m.Input.Bias
	out["flow_lm.flow_net.cond_embed.weight"], out["flow_lm.flow_net.cond_embed.bias"] = m.Condition.Weight, m.Condition.Bias
	for i, v := range m.Time {
		p := fmt.Sprintf("flow_lm.flow_net.time_embed.%d.mlp", i)
		out[p+".0.weight"], out[p+".0.bias"] = v.FC1.Weight, v.FC1.Bias
		out[p+".2.weight"], out[p+".2.bias"] = v.FC2.Weight, v.FC2.Bias
		out[p+".3.alpha"] = v.RMSWeight
	}
	for i, b := range m.Blocks {
		p := fmt.Sprintf("flow_lm.flow_net.res_blocks.%d", i)
		out[p+".in_ln.weight"], out[p+".in_ln.bias"] = b.NormWeight, b.NormBias
		out[p+".mlp.0.weight"], out[p+".mlp.0.bias"] = b.FC1.Weight, b.FC1.Bias
		out[p+".mlp.2.weight"], out[p+".mlp.2.bias"] = b.FC2.Weight, b.FC2.Bias
		out[p+".adaLN_modulation.1.weight"], out[p+".adaLN_modulation.1.bias"] = b.Modulation.Weight, b.Modulation.Bias
	}
	out["flow_lm.flow_net.final_layer.linear.weight"], out["flow_lm.flow_net.final_layer.linear.bias"] = m.Final.Linear.Weight, m.Final.Linear.Bias
	out["flow_lm.flow_net.final_layer.adaLN_modulation.1.weight"], out["flow_lm.flow_net.final_layer.adaLN_modulation.1.bias"] = m.Final.Modulation.Weight, m.Final.Modulation.Bias
}

func addFlowLMGradients(out map[string][]float32, g *FlowLMTrainingGradients) {
	out["flow_lm.conditioner.embed.weight"] = g.Embedding
	out["flow_lm.bos_emb"] = g.BOS
	out["flow_lm.bos_before_voice"] = g.BOSBeforeVoice
	out["flow_lm.speaker_proj_weight"] = g.SpeakerProjection.Weight
	out["flow_lm.input_linear.weight"] = g.Input.Weight
	out["flow_lm.out_eos.weight"], out["flow_lm.out_eos.bias"] = g.EOS.Weight, g.EOS.Bias
	for i, l := range g.Transformer.Layers {
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
	out["flow_lm.out_norm.weight"], out["flow_lm.out_norm.bias"] = g.Transformer.FinalWeight, g.Transformer.FinalBias
}
func addFlowHeadGradientParameters(out map[string][]float32, g *FlowHeadGradients) {
	out["flow_lm.flow_net.input_proj.weight"], out["flow_lm.flow_net.input_proj.bias"] = g.Input.Weight, g.Input.Bias
	out["flow_lm.flow_net.cond_embed.weight"], out["flow_lm.flow_net.cond_embed.bias"] = g.Condition.Weight, g.Condition.Bias
	for i, v := range g.Time {
		p := fmt.Sprintf("flow_lm.flow_net.time_embed.%d.mlp", i)
		out[p+".0.weight"], out[p+".0.bias"] = v.FC1.Weight, v.FC1.Bias
		out[p+".2.weight"], out[p+".2.bias"] = v.FC2.Weight, v.FC2.Bias
		out[p+".3.alpha"] = v.RMSWeight
	}
	for i, b := range g.Blocks {
		p := fmt.Sprintf("flow_lm.flow_net.res_blocks.%d", i)
		out[p+".in_ln.weight"], out[p+".in_ln.bias"] = b.NormWeight, b.NormBias
		out[p+".mlp.0.weight"], out[p+".mlp.0.bias"] = b.FC1.Weight, b.FC1.Bias
		out[p+".mlp.2.weight"], out[p+".mlp.2.bias"] = b.FC2.Weight, b.FC2.Bias
		out[p+".adaLN_modulation.1.weight"], out[p+".adaLN_modulation.1.bias"] = b.Modulation.Weight, b.Modulation.Bias
	}
	out["flow_lm.flow_net.final_layer.linear.weight"], out["flow_lm.flow_net.final_layer.linear.bias"] = g.Final.Linear.Weight, g.Final.Linear.Bias
	out["flow_lm.flow_net.final_layer.adaLN_modulation.1.weight"], out["flow_lm.flow_net.final_layer.adaLN_modulation.1.bias"] = g.Final.Modulation.Weight, g.Final.Modulation.Bias
}
