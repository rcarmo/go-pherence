package pockettts

import (
	"fmt"
	"math"
)

// LSDDistillTrainingBackend composes the explicit flow executor with the
// minimal-stop-gradient LSD objective. Device flow math is separated from the
// small CPU interpolation/loss reductions. Endpoint direct parameter/condition/
// time derivatives are discarded; only its input VJP returns to primary flow.
// No finite-difference approximation replaces time-JVP/mixed derivatives.
func (m *FlowHeadCPU) LSDDistillTrainingBackend(backend TrainingFlowBackend, condition []float32, s, t float32, noise, target []float32, logVariance float32) (LSDDistillResult, LSDDistillGradients, error) {
	return m.lsdDistillTrainingBackendScaled(backend, condition, s, t, noise, target, logVariance, 1)
}
func (m *FlowHeadCPU) lsdDistillTrainingBackendScaled(backend TrainingFlowBackend, condition []float32, s, t float32, noise, target []float32, logVariance, parameterScale float32) (LSDDistillResult, LSDDistillGradients, error) {
	return m.lsdDistillTrainingBackendResults(backend, condition, s, t, noise, target, logVariance, parameterScale, false)
}
func (m *FlowHeadCPU) lsdDistillTrainingStepBackendScaled(backend TrainingFlowBackend, condition []float32, s, t float32, noise, target []float32, logVariance, parameterScale float32) (LSDDistillResult, LSDDistillGradients, error) {
	return m.lsdDistillTrainingBackendResults(backend, condition, s, t, noise, target, logVariance, parameterScale, true)
}
func (m *FlowHeadCPU) lsdDistillTrainingBackendResults(backend TrainingFlowBackend, condition []float32, s, t float32, noise, target []float32, logVariance, parameterScale float32, stepOnly bool) (LSDDistillResult, LSDDistillGradients, error) {
	var result LSDDistillResult
	var gradients LSDDistillGradients
	if backend == nil || m == nil || len(m.Time) != 2 || m.Final.Linear.Out != len(noise) || len(noise) == 0 || len(noise) != len(target) || len(noise) != m.Input.In || !isFinite(s) || !isFinite(t) || s < 0 || t > 1 || s > t || !isFinite(logVariance) || !finiteF32(condition) || !finiteF32(noise) || !finiteF32(target) {
		return result, gradients, fmt.Errorf("invalid backend LSD row")
	}
	n := len(noise)
	branch := m.BranchBackend
	xs := make([]float32, n)
	var primary, endpoint FlowDualTrainingResult
	var err error
	zero := make([]float32, n)
	var xt, dxdt []float32
	delta := t - s
	forwardFusion, forwardFused := backend.(TrainingDistillForwardBackend)
	forwardFused = forwardFused && forwardFusion.DistillForwardEnabled()
	if forwardFused {
		in := DistillForwardInputs{Condition: condition, Target: target, Noise: noise, S: s, T: t}
		var r DistillForwardResult
		var e error
		if admitted, ok := backend.(trainingDistillAdmissionBackend); stepOnly && ok && admitted.distillAdmissionEnabled() {
			r, e = admitted.evaluateDistillAdmitted(m, in, logVariance)
		} else {
			r, e = forwardFusion.EvaluateDistillForward(m, in)
		}
		if e != nil {
			return result, gradients, e
		}
		if e = validateDistillForwardResult(r, in); e != nil {
			return result, gradients, e
		}
		xs, xt, dxdt = r.PrimaryInput, r.EndpointInput, r.Desired
		primary.Output, primary.Tangent, endpoint.Output = r.Velocity, r.Tangent, r.Prediction
	} else {
		if branch != nil {
			values, e := evaluateTrainingBranch(branch, m, "interpolate", [][]float32{target, noise}, s)
			if e != nil {
				return result, gradients, e
			}
			xs = values[0]
		} else {
			for i := range xs {
				xs[i] = s*target[i] + (1-s)*noise[i]
			}
		}
		if err := validateTrainableFlowHead(m, condition, []float32{s, t}, xs, zero); err != nil {
			return result, gradients, err
		}
		primary, err = evaluateTrainingFlow(backend, m, condition, []float32{s, t}, xs, 1, zero, zero, false, 0)
		if err != nil {
			return result, gradients, err
		}
		if err := validateFlowBackendResult(m, primary); err != nil {
			return result, gradients, err
		}
		xt, dxdt = make([]float32, n), make([]float32, n)
		if branch != nil {
			values, e := evaluateTrainingBranch(branch, m, "advance", [][]float32{xs, primary.Output, primary.Tangent}, delta)
			if e != nil {
				return result, gradients, e
			}
			xt, dxdt = values[0], values[1]
		} else {
			for i := range xt {
				xt[i] = xs[i] + delta*primary.Output[i]
				dxdt[i] = primary.Output[i] + delta*primary.Tangent[i]
			}
		}
		endpoint, err = evaluateTrainingFlow(backend, m, condition, []float32{t, t}, xt, -1, zero, zero, false, 0)
		if err != nil {
			return result, gradients, err
		}
		if err := validateFlowBackendResult(m, endpoint); err != nil {
			return result, gradients, err
		}
	}
	fusionDistill, combined := backend.(TrainingDistillBackend)
	combined = combined && fusionDistill.DistillEnabled()
	// The private step consumes raw statistics/condition/target only. Admit and
	// execute the same full graphs before returning, without materialising four
	// owned public diagnostics which the step immediately discards.
	if private, ok := backend.(*ResidentTrainingBackend); stepOnly && combined && ok && private.distillResultsEnabled() {
		in := DistillTrainingInputs{Condition: condition, PrimaryInput: xs, EndpointInput: xt, Prediction: endpoint.Output, Desired: dxdt, Velocity: primary.Output, Tangent: primary.Tangent, Target: target, Noise: noise, S: s, T: t, LogVariance: logVariance}
		r, e := private.evaluateDistillResults(m, in, parameterScale)
		if e != nil {
			return LSDDistillResult{}, gradients, e
		}
		return LSDDistillResult{RawSquare: r.RawSquare, Loss: r.Loss}, LSDDistillGradients{Condition: r.DCondition, Target: r.DTarget, DLogVariance: r.DLogVariance}, nil
	}
	result = LSDDistillResult{Velocity: append([]float32(nil), primary.Output...), TimeDerivative: append([]float32(nil), primary.Tangent...), Endpoint: append([]float32(nil), endpoint.Output...), Residual: make([]float32, n)}
	square := float32(0)
	for i := range result.Residual {
		result.Residual[i] = dxdt[i] - endpoint.Output[i]
		if m.LossBackend == nil {
			square += result.Residual[i] * result.Residual[i]
		}
	}
	if combined {
		in := DistillTrainingInputs{Condition: condition, PrimaryInput: xs, EndpointInput: xt, Prediction: endpoint.Output, Desired: dxdt, Velocity: primary.Output, Tangent: primary.Tangent, Target: target, Noise: noise, S: s, T: t, LogVariance: logVariance}
		r, e := fusionDistill.EvaluateDistill(m, in, parameterScale)
		if e != nil {
			return LSDDistillResult{}, gradients, e
		}
		if e = validateDistillResult(r, in); e != nil {
			return LSDDistillResult{}, gradients, e
		}
		result.RawSquare, result.Loss = r.RawSquare, r.Loss
		return result, LSDDistillGradients{Condition: r.DCondition, Target: r.DTarget, Noise: r.DNoise, DS: r.DTimes[0], DT: r.DTimes[1], DLogVariance: r.DLogVariance}, nil
	}
	var endpointVJP FlowDualTrainingResult
	gv, gt := make([]float32, n), make([]float32, n)
	ddelta := float32(0)
	var dLogvar float32
	fusion, fused := backend.(TrainingEndpointBackend)
	fused = fused && fusion.EndpointEnabled()
	if fused {
		r, e := fusion.EvaluateEndpoint(m, condition, t, xt, endpoint.Output, dxdt, primary.Output, primary.Tangent, logVariance, delta)
		if e != nil {
			return LSDDistillResult{}, gradients, e
		}
		if e = validateEndpointResult(r, condition, xt, endpoint.Output, dxdt, primary.Output, primary.Tangent); e != nil {
			return LSDDistillResult{}, gradients, e
		}
		result.RawSquare, result.Loss, dLogvar = r.RawSquare, r.Loss, r.DLogVariance
		endpointVJP.DInput, gv, gt, ddelta = r.DInput, r.GV, r.GT, r.DDelta
	} else {
		var dr []float32
		if m.LossBackend != nil {
			r, e := m.LossBackend.EvaluateLSDLoss(m, dxdt, endpoint.Output, logVariance)
			if e != nil {
				return LSDDistillResult{}, gradients, e
			}
			if e := validateLSDLossBoundary(r, dxdt, endpoint.Output); e != nil {
				return LSDDistillResult{}, gradients, e
			}
			result.RawSquare, result.Loss, dr, dLogvar = r.RawSquare, r.Loss, r.DPrediction, r.DLogVariance
		} else {
			scale := float32(math.Exp(float64(logVariance))) / float32(n)
			result.RawSquare, result.Loss = float64(square), float64(square*scale-logVariance)
			if !finite64(result.Loss) || !isFinite(scale) {
				return LSDDistillResult{}, gradients, fmt.Errorf("nonfinite backend LSD loss")
			}
			dr = make([]float32, n)
			for i := range dr {
				dr[i] = 2 * result.Residual[i] * scale
			}
			dLogvar = square*scale - 1
		}
		de := make([]float32, n)
		if branch != nil {
			values, e := evaluateTrainingBranch(branch, m, "negate", [][]float32{dr}, 0)
			if e != nil {
				return LSDDistillResult{}, gradients, e
			}
			de = values[0]
		} else {
			for i := range de {
				de[i] = -dr[i]
			}
		}
		endpointVJP, err = evaluateTrainingFlow(backend, m, condition, []float32{t, t}, xt, -1, de, zero, false, 0)
		if err != nil {
			return LSDDistillResult{}, gradients, err
		}
		if err := validateFlowBackendResult(m, endpointVJP); err != nil {
			return LSDDistillResult{}, gradients, err
		}
		if branch != nil {
			values, e := evaluateTrainingBranch(branch, m, "mixed", [][]float32{dr, endpointVJP.DInput, primary.Output, primary.Tangent}, delta)
			if e != nil {
				return LSDDistillResult{}, gradients, e
			}
			gv, gt, ddelta = values[0], values[1], values[2][0]
		} else {
			for i := range gv {
				gv[i] = dr[i] + delta*endpointVJP.DInput[i]
				gt[i] = delta * dr[i]
				ddelta += dr[i]*primary.Tangent[i] + endpointVJP.DInput[i]*primary.Output[i]
			}
		}
	}
	primaryFusion, primaryFused := backend.(TrainingPrimaryBackend)
	primaryFused = primaryFused && primaryFusion.PrimaryEnabled()
	if primaryFused {
		in := PrimaryTrainingInputs{Condition: condition, Input: xs, GV: gv, GT: gt, EndpointInput: endpointVJP.DInput, Target: target, Noise: noise, S: s, T: t, DDelta: ddelta}
		r, e := primaryFusion.EvaluatePrimary(m, in, parameterScale)
		if e != nil {
			return LSDDistillResult{}, gradients, e
		}
		if e = validatePrimaryResult(r, in); e != nil {
			return LSDDistillResult{}, gradients, e
		}
		return result, LSDDistillGradients{Condition: r.DCondition, Target: r.DTarget, Noise: r.DNoise, DS: r.DTimes[0], DT: r.DTimes[1], DLogVariance: dLogvar}, nil
	}
	mixed, err := evaluateTrainingFlow(backend, m, condition, []float32{s, t}, xs, 1, gv, gt, true, parameterScale)
	if err != nil {
		return LSDDistillResult{}, gradients, err
	}
	if err := validateFlowBackendResult(m, mixed); err != nil {
		return LSDDistillResult{}, gradients, err
	}
	var flow *FlowHeadGradients
	if _, resident := backend.(TrainingFlowAccumulator); !resident {
		flow, err = flowGradientsFromNamed(m, mixed.Parameters)
		if err != nil {
			return LSDDistillResult{}, gradients, err
		}
	}
	gradients = LSDDistillGradients{Flow: flow, Condition: append([]float32(nil), mixed.DCondition...), Noise: make([]float32, n), Target: make([]float32, n), DLogVariance: dLogvar}
	if branch != nil {
		values, e := evaluateTrainingBranch(branch, m, "finish", [][]float32{mixed.DInput, endpointVJP.DInput, target, noise, mixed.DTimes, {ddelta}}, s)
		if e != nil {
			return LSDDistillResult{}, LSDDistillGradients{}, e
		}
		gradients.Target, gradients.Noise, gradients.DS, gradients.DT = values[0], values[1], values[2][0], values[2][1]
	} else {
		for i := range xs {
			dx := mixed.DInput[i] + endpointVJP.DInput[i]
			gradients.Target[i] = s * dx
			gradients.Noise[i] = (1 - s) * dx
			gradients.DS += dx * (target[i] - noise[i])
		}
		gradients.DS += mixed.DTimes[0] - ddelta
		gradients.DT = mixed.DTimes[1] + ddelta
	}
	return result, gradients, nil
}
func validateFlowBackendResult(m *FlowHeadCPU, r FlowDualTrainingResult) error {
	if len(r.Output) != m.Final.Linear.Out || len(r.Tangent) != m.Final.Linear.Out || len(r.DCondition) != m.Condition.In || len(r.DInput) != m.Input.In || len(r.DTimes) != len(m.Time) {
		return fmt.Errorf("backend flow result shape")
	}
	for _, values := range [][]float32{r.Output, r.Tangent, r.DCondition, r.DInput, r.DTimes} {
		if !finiteF32(values) {
			return fmt.Errorf("nonfinite backend flow result")
		}
	}
	return nil
}
func accumulateFlowTrainingGradients(dst, src *FlowHeadGradients, scale float32) {
	a, b := map[string][]float32{}, map[string][]float32{}
	addFlowHeadGradientParameters(a, dst)
	addFlowHeadGradientParameters(b, src)
	for name, values := range a {
		for i, x := range b[name] {
			values[i] += x * scale
		}
	}
}
func flowGradientsFromNamed(m *FlowHeadCPU, named map[string][]float32) (*FlowHeadGradients, error) {
	g := newFlowHeadGradients(m)
	bindings := map[string][]float32{}
	addFlowHeadGradientParameters(bindings, g)
	if len(bindings) != len(named) {
		return nil, fmt.Errorf("backend flow gradient name count mismatch")
	}
	for name, dst := range bindings {
		src, ok := named[name]
		if !ok || len(src) != len(dst) || !finiteF32(src) {
			return nil, fmt.Errorf("backend flow gradient %s missing/shape/nonfinite", name)
		}
		copy(dst, src)
	}
	return g, nil
}
