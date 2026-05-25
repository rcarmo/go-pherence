package config

import "fmt"

// Trellis2FlowSchedule mirrors the timestep schedule used by upstream
// trellis2/pipelines/samplers/flow_euler.py. Timesteps are descending from 1 to
// sigma_min and model inputs are scaled by 1000 before inference.
type Trellis2FlowSchedule struct {
	SigmaMin        float64   `json:"sigma_min"`
	Timesteps       []float64 `json:"timesteps"`
	ModelTimesteps  []float64 `json:"model_timesteps"`
	TransitionCount int       `json:"transition_count"`
}

func Trellis2FlowEulerSchedule(steps int, sigmaMin float64) (Trellis2FlowSchedule, error) {
	if steps <= 0 {
		return Trellis2FlowSchedule{}, fmt.Errorf("trellis2 flow schedule: invalid steps %d", steps)
	}
	if sigmaMin < 0 || sigmaMin > 1 {
		return Trellis2FlowSchedule{}, fmt.Errorf("trellis2 flow schedule: invalid sigma_min %g", sigmaMin)
	}
	timesteps := make([]float64, steps)
	if steps == 1 {
		timesteps[0] = 1
	} else {
		for i := 0; i < steps; i++ {
			timesteps[i] = 1 - float64(i)*(1-sigmaMin)/float64(steps-1)
		}
	}
	modelTimesteps := make([]float64, steps)
	for i, t := range timesteps {
		modelTimesteps[i] = 1000 * t
	}
	return Trellis2FlowSchedule{SigmaMin: sigmaMin, Timesteps: timesteps, ModelTimesteps: modelTimesteps, TransitionCount: steps}, nil
}

// Trellis2FlowEulerStep applies upstream sample_once math for a full tensor:
// x_prev = x_t - (t - t_prev) * pred_v.
func Trellis2FlowEulerStep(dst, xT, predV []float32, t, tPrev float64) error {
	if len(dst) != len(xT) || len(predV) != len(xT) {
		return fmt.Errorf("trellis2 flow step: shape mismatch dst=%d x_t=%d pred_v=%d", len(dst), len(xT), len(predV))
	}
	delta := float32(t - tPrev)
	for i := range xT {
		dst[i] = xT[i] - delta*predV[i]
	}
	return nil
}

func Trellis2VToXStartEps(xT, predV []float32, t float64) (x0, eps []float32, err error) {
	if len(xT) != len(predV) {
		return nil, nil, fmt.Errorf("trellis2 v conversion: shape mismatch x_t=%d pred_v=%d", len(xT), len(predV))
	}
	x0 = make([]float32, len(xT))
	eps = make([]float32, len(xT))
	Trellis2VToXStartEpsInto(x0, eps, xT, predV, t)
	return x0, eps, nil
}

// Trellis2VToXStartEpsInto mirrors _v_to_xstart_eps from flow_euler.py:
// pred_x_0 = x_t - pred_v * t
// pred_eps = x_t + pred_v * (1 - t)
func Trellis2VToXStartEpsInto(x0, eps, xT, predV []float32, t float64) error {
	if len(x0) != len(xT) || len(eps) != len(xT) || len(predV) != len(xT) {
		return fmt.Errorf("trellis2 v conversion: shape mismatch x0=%d eps=%d x_t=%d pred_v=%d", len(x0), len(eps), len(xT), len(predV))
	}
	tf := float32(t)
	oneMinusT := float32(1 - t)
	for i := range xT {
		x0[i] = xT[i] - predV[i]*tf
		eps[i] = xT[i] + predV[i]*oneMinusT
	}
	return nil
}

// Trellis2CFGBlend mirrors FlowEulerCfgSampler/GuidanceIntervalSampler output
// blending: neg + strength*(pos-neg). The guidance interval decision stays in
// caller code because upstream applies it around model inference.
func Trellis2CFGBlend(dst, neg, pos []float32, strength float32) error {
	if len(dst) != len(neg) || len(pos) != len(neg) {
		return fmt.Errorf("trellis2 cfg blend: shape mismatch dst=%d neg=%d pos=%d", len(dst), len(neg), len(pos))
	}
	for i := range neg {
		dst[i] = neg[i] + strength*(pos[i]-neg[i])
	}
	return nil
}
