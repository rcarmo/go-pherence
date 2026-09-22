package pockettts

import (
	"fmt"
	"math"
)

// LSDDistillResult is one normalized s->t self-distillation row. RawSquare is
// the unnormalized sum of squared residuals used by upstream metrics.
type LSDDistillResult struct {
	Loss, RawSquare float64
	Velocity        []float32
	TimeDerivative  []float32
	Endpoint        []float32
	Residual        []float32
}

// LSDDistillGradients contains the trainable flow-head gradient and gradients
// for the row inputs. DS and DT include interpolation and delta=t-s paths.
type LSDDistillGradients struct {
	Flow         *FlowHeadGradients
	Condition    []float32
	Noise        []float32
	Target       []float32
	DS, DT       float32
	DLogVariance float32
}

// LSDDistillForwardBackward implements upstream's normalized LSD s->t term for
// one row with stopgrad_type="minimal". The endpoint flow contributes only its
// input VJP: its direct parameter, condition and time gradients are discarded,
// matching f_grad_x_only. The primary flow receives exact reverse-over-JVP
// mixed derivatives.
func (m *FlowHeadCPU) LSDDistillForwardBackward(condition []float32, s, t float32, noise, target []float32, logVariance float32) (LSDDistillResult, LSDDistillGradients, error) {
	if m == nil || len(noise) == 0 || len(noise) != len(target) || len(noise) != m.Input.In || !isFinite(s) || !isFinite(t) || !isFinite(logVariance) {
		return LSDDistillResult{}, LSDDistillGradients{}, fmt.Errorf("invalid Pocket TTS LSD distillation row")
	}
	if s < 0 || s > 1 || t < 0 || t > 1 || s > t {
		return LSDDistillResult{}, LSDDistillGradients{}, fmt.Errorf("Pocket TTS LSD times require 0 <= s <= t <= 1")
	}
	for _, values := range [][]float32{condition, noise, target} {
		for _, value := range values {
			if !isFinite(value) {
				return LSDDistillResult{}, LSDDistillGradients{}, fmt.Errorf("Pocket TTS LSD distillation input is non-finite")
			}
		}
	}
	channels := len(noise)
	xs := make([]float32, channels)
	for i := range xs {
		xs[i] = s*target[i] + (1-s)*noise[i]
	}
	primaryTimes := []float32{s, t}
	velocity, derivative, err := m.ForwardTimeJVP(condition, primaryTimes, xs, 1)
	if err != nil {
		return LSDDistillResult{}, LSDDistillGradients{}, err
	}
	delta := t - s
	xt := make([]float32, channels)
	dxdt := make([]float32, channels)
	for i := range xt {
		xt[i] = xs[i] + delta*velocity[i]
		dxdt[i] = velocity[i] + delta*derivative[i]
	}
	endpointTape, endpoint, err := m.forwardTraining(condition, []float32{t, t}, xt)
	if err != nil {
		return LSDDistillResult{}, LSDDistillGradients{}, err
	}
	result := LSDDistillResult{
		Velocity:       append([]float32(nil), velocity...),
		TimeDerivative: append([]float32(nil), derivative...),
		Endpoint:       append([]float32(nil), endpoint...),
		Residual:       make([]float32, channels),
	}
	square := float32(0)
	for i := range result.Residual {
		result.Residual[i] = dxdt[i] - endpoint[i]
		square += result.Residual[i] * result.Residual[i]
	}
	scale := float32(math.Exp(float64(logVariance))) / float32(channels)
	result.RawSquare = float64(square)
	result.Loss = float64(square*scale - logVariance)
	gradients := LSDDistillGradients{
		Condition:    make([]float32, len(condition)),
		Noise:        make([]float32, channels),
		Target:       make([]float32, channels),
		DLogVariance: square*scale - 1,
	}
	dResidual := make([]float32, channels)
	for i := range dResidual {
		dResidual[i] = 2 * result.Residual[i] * scale
	}
	// Endpoint direct gradients are stopped. Retain only dL/dx_t.
	dEndpoint := make([]float32, channels)
	for i := range dEndpoint {
		dEndpoint[i] = -dResidual[i]
	}
	_, _, dEndpointInput, err := m.backwardTraining(endpointTape, dEndpoint, newFlowHeadGradients(m))
	if err != nil {
		return LSDDistillResult{}, LSDDistillGradients{}, err
	}
	// dxdt=v+delta*dvdt and xt=xs+delta*v.
	dVelocity := make([]float32, channels)
	dDerivative := make([]float32, channels)
	dDelta := float32(0)
	for i := range dVelocity {
		dVelocity[i] = dResidual[i] + delta*dEndpointInput[i]
		dDerivative[i] = delta * dResidual[i]
		dDelta += dResidual[i]*derivative[i] + dEndpointInput[i]*velocity[i]
	}
	_, _, flowGradients, dCondition, dPrimaryTimes, dXSPrimary, err := m.ForwardTimeJVPBackward(condition, primaryTimes, xs, 1, dVelocity, dDerivative)
	if err != nil {
		return LSDDistillResult{}, LSDDistillGradients{}, err
	}
	gradients.Flow = flowGradients
	copy(gradients.Condition, dCondition)
	dXS := make([]float32, channels)
	for i := range dXS {
		dXS[i] = dXSPrimary[i] + dEndpointInput[i]
		gradients.Target[i] = s * dXS[i]
		gradients.Noise[i] = (1 - s) * dXS[i]
		gradients.DS += dXS[i] * (target[i] - noise[i])
	}
	gradients.DS += dPrimaryTimes[0] - dDelta
	gradients.DT = dPrimaryTimes[1] + dDelta
	return result, gradients, nil
}
