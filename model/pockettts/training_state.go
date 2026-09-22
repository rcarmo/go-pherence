package pockettts

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
)

// AdamWConfig controls decoupled AdamW. Use DefaultAdamWConfig for the
// released trainer defaults; a zero WeightDecay is valid.
type AdamWConfig struct {
	LearningRate float32 `json:"learning_rate"`
	Beta1        float32 `json:"beta1"`
	Beta2        float32 `json:"beta2"`
	Epsilon      float32 `json:"epsilon"`
	WeightDecay  float32 `json:"weight_decay"`
}

func DefaultAdamWConfig() AdamWConfig {
	return AdamWConfig{LearningRate: 2e-4, Beta1: 0.9, Beta2: 0.95, Epsilon: 1e-8, WeightDecay: 0.1}
}

func (c AdamWConfig) validate() error {
	for _, value := range []float32{c.LearningRate, c.Beta1, c.Beta2, c.Epsilon, c.WeightDecay} {
		if !isFinite(value) {
			return fmt.Errorf("Pocket TTS AdamW config is non-finite")
		}
	}
	if c.LearningRate <= 0 || c.Beta1 < 0 || c.Beta1 >= 1 || c.Beta2 < 0 || c.Beta2 >= 1 || c.Epsilon <= 0 || c.WeightDecay < 0 {
		return fmt.Errorf("invalid Pocket TTS AdamW config")
	}
	return nil
}

// NamedTrainingTensor is a deterministic checkpoint tensor representation.
type NamedTrainingTensor struct {
	Name   string    `json:"name"`
	Values []float32 `json:"values"`
}

// NamedAdamState stores AdamW moments for one parameter.
type NamedAdamState struct {
	Name string    `json:"name"`
	M    []float32 `json:"m"`
	V    []float32 `json:"v"`
}

// TrainingState is the versioned, non-executable resumable state for the tiny
// parity topology. Released safetensors export is a later full-model boundary.
type TrainingState struct {
	Version    int                   `json:"version"`
	Step       int                   `json:"step"`
	HiddenDim  int                   `json:"hidden_dim"`
	LatentDim  int                   `json:"latent_dim"`
	AdamW      AdamWConfig           `json:"adamw"`
	Parameters []NamedTrainingTensor `json:"parameters"`
	Adam       []NamedAdamState      `json:"adam"`
	EMA        []NamedTrainingTensor `json:"ema,omitempty"`
	EMADecay   float32               `json:"ema_decay,omitempty"`
}

// TinyTrainer owns optimizer and EMA state. Its model must not be used from
// another goroutine while Step mutates it.
type TinyTrainer struct {
	Model     *TinyTrainingModel
	Config    AdamWConfig
	StepCount int
	Moments   map[string]NamedAdamState
	EMA       map[string][]float32
	EMADecay  float32
}

func NewTinyTrainer(model *TinyTrainingModel, config AdamWConfig, emaDecay float32) (*TinyTrainer, error) {
	if model == nil {
		return nil, fmt.Errorf("Pocket TTS tiny training model is nil")
	}
	if err := model.Validate(); err != nil {
		return nil, err
	}
	if err := config.validate(); err != nil {
		return nil, err
	}
	if !isFinite(emaDecay) || emaDecay < 0 || emaDecay >= 1 {
		return nil, fmt.Errorf("Pocket TTS EMA decay must be in [0,1)")
	}
	trainer := &TinyTrainer{Model: model, Config: config, Moments: make(map[string]NamedAdamState), EMADecay: emaDecay}
	if emaDecay > 0 {
		trainer.EMA = cloneTrainingMap(model.trainingParameterMap())
	}
	return trainer, nil
}

// Step applies decoupled AdamW and then EMA, matching upstream ordering.
func (t *TinyTrainer) Step(grads TinyTrainingGradients) error {
	if t == nil || t.Model == nil {
		return fmt.Errorf("Pocket TTS trainer is nil")
	}
	if err := t.Model.Validate(); err != nil {
		return err
	}
	config := t.Config
	if err := config.validate(); err != nil {
		return err
	}
	params := t.Model.trainingParameterMap()
	// Copy the public gradient input so even caller-supplied slices that alias
	// model parameters cannot change underneath the in-place optimiser update.
	gradientMap := cloneTrainingMap(grads.parameterMap())
	for name, values := range params {
		gradient, ok := gradientMap[name]
		if !ok || len(gradient) != len(values) {
			return fmt.Errorf("Pocket TTS gradient shape mismatch for %q", name)
		}
		for _, g := range gradient {
			if !isFinite(g) {
				return fmt.Errorf("Pocket TTS gradient %q is non-finite", name)
			}
		}
		if state, ok := t.Moments[name]; ok && (len(state.M) != len(values) || len(state.V) != len(values)) {
			return fmt.Errorf("Pocket TTS AdamW state shape mismatch for %q", name)
		}
		if t.EMADecay > 0 && t.EMA != nil && len(t.EMA[name]) != len(values) {
			return fmt.Errorf("Pocket TTS EMA state shape mismatch for %q", name)
		}
	}
	step := t.StepCount + 1
	b1Correction := float32(1 - math.Pow(float64(config.Beta1), float64(step)))
	b2Correction := float32(1 - math.Pow(float64(config.Beta2), float64(step)))
	for _, name := range sortedTrainingKeys(params) {
		values, gradient := params[name], gradientMap[name]
		state, ok := t.Moments[name]
		if !ok {
			state = NamedAdamState{Name: name, M: make([]float32, len(values)), V: make([]float32, len(values))}
		}
		for i := range values {
			g := gradient[i]
			state.M[i] = config.Beta1*state.M[i] + (1-config.Beta1)*g
			state.V[i] = config.Beta2*state.V[i] + (1-config.Beta2)*g*g
			mHat := state.M[i] / b1Correction
			vHat := state.V[i] / b2Correction
			values[i] *= 1 - config.LearningRate*config.WeightDecay
			values[i] -= config.LearningRate * mHat / (float32(math.Sqrt(float64(vHat))) + config.Epsilon)
		}
		t.Moments[name] = state
	}
	// Scalar fields travel through one-element slices; copy them back.
	t.Model.FlowLogVariance = params["flow.logvar"][0]
	t.Model.EOSBias = params["eos.bias"][0]
	t.StepCount = step
	if t.EMADecay > 0 {
		if t.EMA == nil {
			t.EMA = cloneTrainingMap(params)
		} else {
			for name, values := range params {
				shadow := t.EMA[name]
				for i := range values {
					shadow[i] = t.EMADecay*shadow[i] + (1-t.EMADecay)*values[i]
				}
			}
		}
	}
	return nil
}

func (t *TinyTrainer) State() (TrainingState, error) {
	if t == nil || t.Model == nil {
		return TrainingState{}, fmt.Errorf("Pocket TTS trainer is nil")
	}
	if err := t.Model.Validate(); err != nil {
		return TrainingState{}, err
	}
	config := t.Config
	if err := config.validate(); err != nil {
		return TrainingState{}, err
	}
	state := TrainingState{Version: 1, Step: t.StepCount, HiddenDim: t.Model.HiddenDim, LatentDim: t.Model.LatentDim, AdamW: config, EMADecay: t.EMADecay}
	state.Parameters = trainingMapToList(t.Model.trainingParameterMap())
	for _, name := range sortedAdamKeys(t.Moments) {
		value := t.Moments[name]
		state.Adam = append(state.Adam, NamedAdamState{Name: name, M: append([]float32(nil), value.M...), V: append([]float32(nil), value.V...)})
	}
	if t.EMADecay > 0 {
		state.EMA = trainingMapToList(t.EMA)
	}
	if err := state.Validate(); err != nil {
		return TrainingState{}, err
	}
	return state, nil
}

func (s TrainingState) Validate() error {
	if s.Version != 1 || s.Step < 0 || s.HiddenDim <= 0 || s.LatentDim <= 0 || !isFinite(s.EMADecay) || s.EMADecay < 0 || s.EMADecay >= 1 {
		return fmt.Errorf("invalid Pocket TTS training state header")
	}
	if err := s.AdamW.validate(); err != nil {
		return fmt.Errorf("Pocket TTS training state AdamW config: %w", err)
	}
	model, err := modelFromTrainingTensors(s.HiddenDim, s.LatentDim, s.Parameters)
	if err != nil {
		return err
	}
	params := model.trainingParameterMap()
	moments, err := adamListToMap(s.Adam, params, s.Step > 0)
	if err != nil {
		return err
	}
	_ = moments
	if s.EMADecay == 0 {
		if len(s.EMA) != 0 {
			return fmt.Errorf("Pocket TTS training state has EMA tensors with zero decay")
		}
	} else if _, err := tensorListToMap(s.EMA, params, "EMA"); err != nil {
		return err
	}
	return nil
}

func NewTinyTrainerFromState(state TrainingState) (*TinyTrainer, error) {
	if err := state.Validate(); err != nil {
		return nil, err
	}
	model, _ := modelFromTrainingTensors(state.HiddenDim, state.LatentDim, state.Parameters)
	trainer, err := NewTinyTrainer(model, state.AdamW, state.EMADecay)
	if err != nil {
		return nil, err
	}
	trainer.StepCount = state.Step
	trainer.Moments, _ = adamListToMap(state.Adam, model.trainingParameterMap(), state.Step > 0)
	if state.EMADecay > 0 {
		trainer.EMA, _ = tensorListToMap(state.EMA, model.trainingParameterMap(), "EMA")
	}
	return trainer, nil
}

func ReadTrainingState(r io.Reader) (TrainingState, error) {
	var state TrainingState
	if r == nil {
		return state, fmt.Errorf("Pocket TTS training state reader is nil")
	}
	decoder := json.NewDecoder(io.LimitReader(r, 256<<20))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&state); err != nil {
		return state, fmt.Errorf("decode Pocket TTS training state: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return state, fmt.Errorf("Pocket TTS training state contains trailing data")
	}
	return state, state.Validate()
}

func LoadTrainingState(path string) (TrainingState, error) {
	f, err := os.Open(path)
	if err != nil {
		return TrainingState{}, err
	}
	defer f.Close()
	return ReadTrainingState(f)
}

// SaveTrainingState writes atomically and fsyncs the temporary file before
// publication so an interrupted save cannot expose partial JSON.
func SaveTrainingState(path string, state TrainingState) error {
	if err := state.Validate(); err != nil {
		return err
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	f, err := os.CreateTemp(dir, ".pockettts-training-*.tmp")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	encoder := json.NewEncoder(f)
	if err = encoder.Encode(state); err != nil {
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
	directory, err := os.Open(dir)
	if err != nil {
		return err
	}
	if err = directory.Sync(); err != nil {
		directory.Close()
		return err
	}
	return directory.Close()
}

func modelFromTrainingTensors(hidden, latent int, values []NamedTrainingTensor) (*TinyTrainingModel, error) {
	model := &TinyTrainingModel{HiddenDim: hidden, LatentDim: latent, FlowWeight: make([]float32, latent*(hidden+latent+1)), FlowBias: make([]float32, latent), EOSWeight: make([]float32, hidden)}
	params := model.trainingParameterMap()
	loaded, err := tensorListToMap(values, params, "parameter")
	if err != nil {
		return nil, err
	}
	copy(model.FlowWeight, loaded["flow.weight"])
	copy(model.FlowBias, loaded["flow.bias"])
	model.FlowLogVariance = loaded["flow.logvar"][0]
	copy(model.EOSWeight, loaded["eos.weight"])
	model.EOSBias = loaded["eos.bias"][0]
	return model, model.Validate()
}

func tensorListToMap(list []NamedTrainingTensor, expected map[string][]float32, label string) (map[string][]float32, error) {
	if len(list) != len(expected) {
		return nil, fmt.Errorf("Pocket TTS %s tensor count=%d want=%d", label, len(list), len(expected))
	}
	out := make(map[string][]float32, len(list))
	for _, tensor := range list {
		want, ok := expected[tensor.Name]
		if !ok || out[tensor.Name] != nil || len(tensor.Values) != len(want) {
			return nil, fmt.Errorf("invalid or duplicate Pocket TTS %s tensor %q", label, tensor.Name)
		}
		for _, value := range tensor.Values {
			if !isFinite(value) {
				return nil, fmt.Errorf("Pocket TTS %s tensor %q is non-finite", label, tensor.Name)
			}
		}
		out[tensor.Name] = append([]float32(nil), tensor.Values...)
	}
	return out, nil
}

func adamListToMap(list []NamedAdamState, expected map[string][]float32, required bool) (map[string]NamedAdamState, error) {
	if !required && len(list) == 0 {
		return make(map[string]NamedAdamState), nil
	}
	if len(list) != len(expected) {
		return nil, fmt.Errorf("Pocket TTS AdamW tensor count=%d want=%d", len(list), len(expected))
	}
	out := make(map[string]NamedAdamState, len(list))
	for _, state := range list {
		want, ok := expected[state.Name]
		if !ok || out[state.Name].Name != "" || len(state.M) != len(want) || len(state.V) != len(want) {
			return nil, fmt.Errorf("invalid or duplicate Pocket TTS AdamW tensor %q", state.Name)
		}
		for _, values := range [][]float32{state.M, state.V} {
			for _, value := range values {
				if !isFinite(value) {
					return nil, fmt.Errorf("Pocket TTS AdamW tensor %q is non-finite", state.Name)
				}
			}
		}
		out[state.Name] = NamedAdamState{Name: state.Name, M: append([]float32(nil), state.M...), V: append([]float32(nil), state.V...)}
	}
	return out, nil
}

func trainingMapToList(values map[string][]float32) []NamedTrainingTensor {
	out := make([]NamedTrainingTensor, 0, len(values))
	for _, name := range sortedTrainingKeys(values) {
		out = append(out, NamedTrainingTensor{Name: name, Values: append([]float32(nil), values[name]...)})
	}
	return out
}

func cloneTrainingMap(values map[string][]float32) map[string][]float32 {
	out := make(map[string][]float32, len(values))
	for name, value := range values {
		out[name] = append([]float32(nil), value...)
	}
	return out
}

func sortedTrainingKeys(values map[string][]float32) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func sortedAdamKeys(values map[string]NamedAdamState) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
