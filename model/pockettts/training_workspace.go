package pockettts

import (
	"fmt"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// TrainingWorkspace owns reusable mutable buffers for one fixed training
// topology and batch shape. It is not safe for concurrent use. Results from
// PocketTrainingStepInto alias this workspace until the next call.
type TrainingWorkspace struct {
	frames, hidden, latent int
	weightRows             int
	flow                   *FlowHeadCPU
	weightInputs           []float32
	zeroZ, zeroEOS         []float32
	dZ, directTarget       []float32
	dEOS                   []float32
	dDiagLogvar            []float32
	dDistillLogvar         []float32
	dLogvars               []float32
	xTime, desired         []float32
	flowGradients          *FlowHeadGradients
	discardFlowGradients   *FlowHeadGradients
}

func NewTrainingWorkspace(flowLM *FlowLMTrainingCPU, flow *FlowHeadCPU, batch FlowLMTrainingBatch) (*TrainingWorkspace, error) {
	if flowLM == nil || flow == nil || batch.Frames <= 0 || flowLM.Hidden <= 0 || flowLM.LatentDim <= 0 {
		return nil, fmt.Errorf("invalid Pocket TTS training workspace")
	}
	frames, hidden, latent := batch.Frames, flowLM.Hidden, flowLM.LatentDim
	weightRows, ok := checked.MulInt(2, frames)
	if !ok {
		return nil, fmt.Errorf("invalid Pocket TTS workspace weight rows")
	}
	weightElements, ok := checked.MulInt(4, frames)
	if !ok {
		return nil, fmt.Errorf("invalid Pocket TTS workspace weight shape")
	}
	hiddenElements, ok := checked.MulInt(frames, hidden)
	if !ok {
		return nil, fmt.Errorf("invalid Pocket TTS workspace hidden shape")
	}
	latentElements, ok := checked.MulInt(frames, latent)
	if !ok {
		return nil, fmt.Errorf("invalid Pocket TTS workspace latent shape")
	}
	w := &TrainingWorkspace{
		frames: frames, hidden: hidden, latent: latent, weightRows: weightRows, flow: flow,
		weightInputs: make([]float32, weightElements), zeroZ: make([]float32, hiddenElements), zeroEOS: make([]float32, frames),
		dZ: make([]float32, hiddenElements), directTarget: make([]float32, latentElements), dEOS: make([]float32, frames),
		dDiagLogvar: make([]float32, frames), dDistillLogvar: make([]float32, frames), dLogvars: make([]float32, weightRows),
		xTime: make([]float32, latent), desired: make([]float32, latent),
		flowGradients: newFlowHeadGradients(flow), discardFlowGradients: newFlowHeadGradients(flow),
	}
	return w, nil
}

func (w *TrainingWorkspace) compatible(flow *FlowHeadCPU) bool {
	if w == nil || flow == nil || w.flow != flow || w.flowGradients == nil || w.discardFlowGradients == nil || len(flow.Time) != len(w.flowGradients.Time) || len(flow.Blocks) != len(w.flowGradients.Blocks) {
		return false
	}
	if !linearGradientMatches(w.flowGradients.Input, flow.Input) || !linearGradientMatches(w.flowGradients.Condition, flow.Condition) || !linearGradientMatches(w.flowGradients.Final.Linear, flow.Final.Linear) || !linearGradientMatches(w.flowGradients.Final.Modulation, flow.Final.Modulation) {
		return false
	}
	for i := range flow.Time {
		if !linearGradientMatches(w.flowGradients.Time[i].FC1, flow.Time[i].FC1) || !linearGradientMatches(w.flowGradients.Time[i].FC2, flow.Time[i].FC2) || len(w.flowGradients.Time[i].RMSWeight) != len(flow.Time[i].RMSWeight) {
			return false
		}
	}
	for i := range flow.Blocks {
		if len(w.flowGradients.Blocks[i].NormWeight) != len(flow.Blocks[i].NormWeight) || len(w.flowGradients.Blocks[i].NormBias) != len(flow.Blocks[i].NormBias) || !linearGradientMatches(w.flowGradients.Blocks[i].FC1, flow.Blocks[i].FC1) || !linearGradientMatches(w.flowGradients.Blocks[i].FC2, flow.Blocks[i].FC2) || !linearGradientMatches(w.flowGradients.Blocks[i].Modulation, flow.Blocks[i].Modulation) {
			return false
		}
	}
	return true
}

func linearGradientMatches(gradient LinearF32Gradient, linear LinearF32) bool {
	return len(gradient.Weight) == len(linear.Weight) && len(gradient.Bias) == len(linear.Bias)
}

func (w *TrainingWorkspace) reset() {
	clear(w.weightInputs)
	clear(w.zeroZ)
	clear(w.zeroEOS)
	clear(w.dZ)
	clear(w.directTarget)
	clear(w.dEOS)
	clear(w.dDiagLogvar)
	clear(w.dDistillLogvar)
	clear(w.dLogvars)
	clear(w.xTime)
	clear(w.desired)
	clearFlowHeadGradients(w.flowGradients)
	clearFlowHeadGradients(w.discardFlowGradients)
}

func clearFlowHeadGradients(g *FlowHeadGradients) {
	if g == nil {
		return
	}
	clearLinearGradient(&g.Input)
	clearLinearGradient(&g.Condition)
	for i := range g.Time {
		clearLinearGradient(&g.Time[i].FC1)
		clearLinearGradient(&g.Time[i].FC2)
		clear(g.Time[i].RMSWeight)
	}
	for i := range g.Blocks {
		clear(g.Blocks[i].NormWeight)
		clear(g.Blocks[i].NormBias)
		clearLinearGradient(&g.Blocks[i].FC1)
		clearLinearGradient(&g.Blocks[i].FC2)
		clearLinearGradient(&g.Blocks[i].Modulation)
	}
	clearLinearGradient(&g.Final.Linear)
	clearLinearGradient(&g.Final.Modulation)
}
func clearLinearGradient(g *LinearF32Gradient) { clear(g.Weight); clear(g.Bias) }
