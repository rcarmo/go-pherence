package pockettts

import "fmt"

// TinyTrainingBatch is a deterministic frozen-backbone training boundary. It
// deliberately excludes transformer backward: Hidden is the frozen FlowLM
// output and only the affine Flow/EOS heads below are trainable. Arrays are
// row-major; Mask selects valid latent frames.
type TinyTrainingBatch struct {
	Rows, HiddenDim, LatentDim int
	Hidden                     []float32
	Noise, Target              []float32
	Times                      []float32
	Mask                       []bool
}

// TinyTrainingModel is the first native gradient-parity topology. FlowInput
// concatenates [hidden, x_t, t], and FlowWeight is [latent, input]. EOSWeight
// is [hidden]. FlowLogVariance represents the normalized LSD diagonal weight.
type TinyTrainingModel struct {
	HiddenDim, LatentDim int
	FlowWeight           []float32
	FlowBias             []float32
	FlowLogVariance      float32
	EOSWeight            []float32
	EOSBias              float32
}

// TinyTrainingConfig mirrors the top-level upstream reduction weights. Use
// DefaultTinyTrainingConfig for the released defaults; zero weights are valid.
type TinyTrainingConfig struct {
	PEqual        float32
	EOSLossWeight float32
}

// TinyTrainingMetrics separates the two objectives from their weighted sum.
type TinyTrainingMetrics struct {
	FlowDiagonal float64
	EOS          float64
	Loss         float64
}

// TinyTrainingGradients has the same shapes as TinyTrainingModel.
type TinyTrainingGradients struct {
	FlowWeight      []float32
	FlowBias        []float32
	FlowLogVariance float32
	EOSWeight       []float32
	EOSBias         float32
}

func (m *TinyTrainingModel) Validate() error {
	if m == nil || m.HiddenDim <= 0 || m.LatentDim <= 0 {
		return fmt.Errorf("invalid Pocket TTS tiny training model")
	}
	flowIn := m.HiddenDim + m.LatentDim + 1
	if len(m.FlowWeight) != m.LatentDim*flowIn || len(m.FlowBias) != m.LatentDim || len(m.EOSWeight) != m.HiddenDim {
		return fmt.Errorf("invalid Pocket TTS tiny training parameter shape")
	}
	for _, values := range [][]float32{m.FlowWeight, m.FlowBias, m.EOSWeight, {m.FlowLogVariance, m.EOSBias}} {
		for _, value := range values {
			if !isFinite(value) {
				return fmt.Errorf("Pocket TTS tiny training parameter is non-finite")
			}
		}
	}
	return nil
}

func (b TinyTrainingBatch) validate(model *TinyTrainingModel) error {
	if b.Rows <= 0 || b.HiddenDim != model.HiddenDim || b.LatentDim != model.LatentDim || len(b.Hidden) != b.Rows*b.HiddenDim || len(b.Noise) != b.Rows*b.LatentDim || len(b.Target) != len(b.Noise) || len(b.Times) != b.Rows || len(b.Mask) != b.Rows {
		return fmt.Errorf("invalid Pocket TTS tiny training batch shape")
	}
	valid := 0
	for row := 0; row < b.Rows; row++ {
		if !isFinite(b.Times[row]) || b.Times[row] < 0 || b.Times[row] > 1 {
			return fmt.Errorf("Pocket TTS tiny training time %d is outside [0,1]", row)
		}
		if b.Mask[row] {
			valid++
		}
	}
	if valid == 0 {
		return fmt.Errorf("Pocket TTS tiny training batch has no valid latent frames")
	}
	for _, values := range [][]float32{b.Hidden, b.Noise, b.Target} {
		for _, value := range values {
			if !isFinite(value) {
				return fmt.Errorf("Pocket TTS tiny training batch is non-finite")
			}
		}
	}
	return nil
}

func DefaultTinyTrainingConfig() TinyTrainingConfig {
	return TinyTrainingConfig{PEqual: 0.75, EOSLossWeight: 0.1}
}

func (c TinyTrainingConfig) validate() error {
	if !isFinite(c.PEqual) || c.PEqual < 0 || c.PEqual > 1 || !isFinite(c.EOSLossWeight) || c.EOSLossWeight < 0 {
		return fmt.Errorf("invalid Pocket TTS tiny training loss weights")
	}
	return nil
}

// ForwardBackwardTinyTraining evaluates the frozen topology and returns exact
// analytic gradients. The full LSD s->t/JVP term is intentionally not part of
// this boundary; this is the diagonal objective required before adding it.
func ForwardBackwardTinyTraining(model *TinyTrainingModel, batch TinyTrainingBatch, config TinyTrainingConfig) (TinyTrainingMetrics, TinyTrainingGradients, error) {
	if err := model.Validate(); err != nil {
		return TinyTrainingMetrics{}, TinyTrainingGradients{}, err
	}
	if err := batch.validate(model); err != nil {
		return TinyTrainingMetrics{}, TinyTrainingGradients{}, err
	}
	if err := config.validate(); err != nil {
		return TinyTrainingMetrics{}, TinyTrainingGradients{}, err
	}

	h, c := model.HiddenDim, model.LatentDim
	flowIn := h + c + 1
	validRows := 0
	for _, valid := range batch.Mask {
		if valid {
			validRows++
		}
	}
	inputs := make([]float32, validRows*flowIn)
	prediction := make([]float32, validRows*c)
	desired := make([]float32, validRows*c)
	selected := 0
	for row, valid := range batch.Mask {
		if !valid {
			continue
		}
		in := inputs[selected*flowIn : (selected+1)*flowIn]
		copy(in, batch.Hidden[row*h:(row+1)*h])
		t := batch.Times[row]
		for channel := 0; channel < c; channel++ {
			i := row*c + channel
			in[h+channel] = t*batch.Target[i] + (1-t)*batch.Noise[i]
			desired[selected*c+channel] = batch.Target[i] - batch.Noise[i]
		}
		in[flowIn-1] = t
		for out := 0; out < c; out++ {
			value := model.FlowBias[out]
			weight := model.FlowWeight[out*flowIn : (out+1)*flowIn]
			for i, input := range in {
				value += weight[i] * input
			}
			prediction[selected*c+out] = value
		}
		selected++
	}
	logvar := make([]float32, validRows)
	for i := range logvar {
		logvar[i] = model.FlowLogVariance
	}
	flowLoss, dPrediction, dLogvar, err := LSDDiagonalLossAndGradient(prediction, desired, logvar, c, true)
	if err != nil {
		return TinyTrainingMetrics{}, TinyTrainingGradients{}, err
	}

	eosLogits := make([]float32, batch.Rows)
	for row := 0; row < batch.Rows; row++ {
		value := model.EOSBias
		for i := 0; i < h; i++ {
			value += model.EOSWeight[i] * batch.Hidden[row*h+i]
		}
		eosLogits[row] = value
	}
	eosLoss, dEOS, err := EOSLossAndGradient(eosLogits, batch.Mask)
	if err != nil {
		return TinyTrainingMetrics{}, TinyTrainingGradients{}, err
	}

	grads := TinyTrainingGradients{
		FlowWeight: make([]float32, len(model.FlowWeight)),
		FlowBias:   make([]float32, len(model.FlowBias)),
		EOSWeight:  make([]float32, len(model.EOSWeight)),
	}
	for selected := 0; selected < validRows; selected++ {
		in := inputs[selected*flowIn : (selected+1)*flowIn]
		for out := 0; out < c; out++ {
			g := config.PEqual * dPrediction[selected*c+out]
			grads.FlowBias[out] += g
			for i, input := range in {
				grads.FlowWeight[out*flowIn+i] += g * input
			}
		}
	}
	for _, g := range dLogvar {
		grads.FlowLogVariance += config.PEqual * g
	}
	for row, g := range dEOS {
		g *= config.EOSLossWeight
		grads.EOSBias += g
		for i := 0; i < h; i++ {
			grads.EOSWeight[i] += g * batch.Hidden[row*h+i]
		}
	}
	metrics := TinyTrainingMetrics{
		FlowDiagonal: flowLoss,
		EOS:          eosLoss,
		Loss:         float64(config.PEqual)*flowLoss + float64(config.EOSLossWeight)*eosLoss,
	}
	return metrics, grads, nil
}

func (m *TinyTrainingModel) trainingParameterMap() map[string][]float32 {
	return map[string][]float32{
		"flow.weight": m.FlowWeight,
		"flow.bias":   m.FlowBias,
		"flow.logvar": {m.FlowLogVariance},
		"eos.weight":  m.EOSWeight,
		"eos.bias":    {m.EOSBias},
	}
}

func (g TinyTrainingGradients) parameterMap() map[string][]float32 {
	return map[string][]float32{
		"flow.weight": g.FlowWeight,
		"flow.bias":   g.FlowBias,
		"flow.logvar": {g.FlowLogVariance},
		"eos.weight":  g.EOSWeight,
		"eos.bias":    {g.EOSBias},
	}
}
