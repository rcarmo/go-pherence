package pockettts

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// TrainingWorkspace owns reusable mutable buffers for one fixed training
// topology and batch shape. It is not safe for concurrent use. Results from
// PocketTrainingStepInto alias this workspace until the next call.
type TrainingWorkspace struct {
	frames, hidden, latent                                                       int
	weightRows                                                                   int
	flowLM                                                                       *FlowLMTrainingCPU
	transformerWidth, transformerHeads, transformerHeadDim, transformerContext   int
	transformerMaxPeriod                                                         float64
	flow                                                                         *FlowHeadCPU
	weightInputs                                                                 []float32
	zeroZ, zeroEOS                                                               []float32
	dZ, directTarget                                                             []float32
	dEOS                                                                         []float32
	dDiagLogvar                                                                  []float32
	dDistillLogvar                                                               []float32
	dLogvars                                                                     []float32
	xTime, desired                                                               []float32
	flowGradients                                                                *FlowHeadGradients
	discardFlowGradients                                                         *FlowHeadGradients
	flowLMGradients                                                              *FlowLMTrainingGradients
	flowLMInputGradients                                                         FlowLMTrainingInputGradients
	primaryFlowTape, endpointFlowTape, primaryFlowBackward, endpointFlowBackward *flowTapeScratch
}

// NewAdmittedTrainingWorkspace allocates only after PlanTrainingShape has
// admitted this exact production row under an explicit resident-byte ceiling.
func NewAdmittedTrainingWorkspace(flowLM *FlowLMTrainingCPU, flow *FlowHeadCPU, weighting *LSDWeightMLP, batch FlowLMTrainingBatch, plan TrainingShapePlan) (*TrainingWorkspace, error) {
	a := plan.admission
	if flowLM == nil || flowLM.Transformer == nil || flow == nil || weighting == nil || a.targetFrames <= 0 || batch.Frames != a.targetFrames || batch.VoiceFrames != a.voiceFrames || len(batch.TextTokens) != a.textTokens || flowLM.Hidden != a.hidden || flowLM.LatentDim != a.latent || flowLM.Vocabulary != a.vocabulary || len(flowLM.Embedding) != a.vocabulary*a.hidden || len(flowLM.BOS) != a.latent || len(flowLM.BOSBeforeVoice) != a.hidden || flowLM.Transformer.Heads != a.heads || flowLM.Transformer.Context != a.transformerContext || flowLM.Transformer.MaxPeriod != a.transformerMaxPeriod || len(flowLM.Transformer.Layers) != a.transformerLayers || len(flow.Time) != 2 || len(flow.Blocks) != a.flowDepth || flow.Input.In != a.latent || flow.Input.Out != a.flowDim || flow.Condition.In != a.hidden || flow.Condition.Out != a.flowDim || flow.Final.Linear.In != a.flowDim || flow.Final.Linear.Out != a.latent {
		return nil, fmt.Errorf("Pocket TTS training batch does not match admitted shape")
	}
	for _, layer := range flowLM.Transformer.Layers {
		hasScale := layer.LayerScale1 != nil || layer.LayerScale2 != nil
		if layer.FC1.Out != a.feedForward || hasScale != a.layerScale || ((layer.LayerScale1 == nil) != (layer.LayerScale2 == nil)) {
			return nil, fmt.Errorf("Pocket TTS training transformer does not match admitted shape")
		}
	}
	for _, time := range flow.Time {
		if time.FC1.In != 256 || time.FC1.Out != a.flowDim || time.FC2.In != a.flowDim || time.FC2.Out != a.flowDim || len(time.RMSWeight) != a.flowDim {
			return nil, fmt.Errorf("Pocket TTS training time embedding does not match admitted shape")
		}
	}
	for _, block := range flow.Blocks {
		if block.FC1.In != a.flowDim || block.FC1.Out != a.flowDim || block.FC2.In != a.flowDim || block.FC2.Out != a.flowDim || block.Modulation.In != a.flowDim || block.Modulation.Out != 3*a.flowDim {
			return nil, fmt.Errorf("Pocket TTS training flow block does not match admitted shape")
		}
	}
	if err := validateOwnedTrainingLinear(flowLM.SpeakerProjection, false); err != nil {
		return nil, err
	}
	if err := validateOwnedTrainingLinear(flowLM.Input, false); err != nil {
		return nil, err
	}
	if err := validateOwnedTrainingLinear(flowLM.EOS, true); err != nil {
		return nil, err
	}
	zeroSequence := make([]float32, a.hidden)
	if err := validateTrainableTransformer(flowLM.Transformer, zeroSequence, zeroSequence, 1); err != nil {
		return nil, err
	}
	if err := validateTrainableFlowHead(flow, make([]float32, a.hidden), make([]float32, 2), make([]float32, a.latent), make([]float32, a.latent)); err != nil {
		return nil, err
	}
	wantWeighting := [4][2]int{{2, 32}, {32, 32}, {32, 32}, {32, 1}}
	if len(weighting.Layers) != len(wantWeighting) {
		return nil, fmt.Errorf("Pocket TTS training weighting does not match admitted shape")
	}
	for i, layer := range weighting.Layers {
		if err := validateOwnedTrainingLinear(layer, true); err != nil {
			return nil, err
		}
		if layer.In != wantWeighting[i][0] || layer.Out != wantWeighting[i][1] {
			return nil, fmt.Errorf("Pocket TTS training weighting does not match admitted shape")
		}
	}
	params, err := fullParameterMap(flowLM, flow, weighting)
	if err != nil {
		return nil, err
	}
	var elements int64
	for _, values := range params {
		if int64(len(values)) > math.MaxInt64-elements {
			return nil, fmt.Errorf("Pocket TTS training parameter count overflows")
		}
		elements += int64(len(values))
	}
	if elements != a.parameterElements {
		return nil, fmt.Errorf("Pocket TTS training parameters do not match admitted shape")
	}
	return NewTrainingWorkspace(flowLM, flow, batch)
}

func NewTrainingWorkspace(flowLM *FlowLMTrainingCPU, flow *FlowHeadCPU, batch FlowLMTrainingBatch) (*TrainingWorkspace, error) {
	if flowLM == nil || flowLM.Transformer == nil || flow == nil || len(flow.Time) != 2 || batch.Frames <= 0 || flowLM.Hidden <= 0 || flowLM.LatentDim <= 0 {
		return nil, fmt.Errorf("invalid Pocket TTS training workspace")
	}
	primaryFlowTape, err := newFlowTapeScratch(flow)
	if err != nil {
		return nil, err
	}
	endpointFlowTape, err := newFlowTapeScratch(flow)
	if err != nil {
		return nil, err
	}
	primaryFlowBackward, err := newFlowTapeScratch(flow)
	if err != nil {
		return nil, err
	}
	endpointFlowBackward, err := newFlowTapeScratch(flow)
	if err != nil {
		return nil, err
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
		frames: frames, hidden: hidden, latent: latent, weightRows: weightRows, flowLM: flowLM,
		transformerWidth: flowLM.Transformer.Width, transformerHeads: flowLM.Transformer.Heads,
		transformerHeadDim: flowLM.Transformer.HeadDim, transformerContext: flowLM.Transformer.Context,
		transformerMaxPeriod: flowLM.Transformer.MaxPeriod, flow: flow,
		weightInputs: make([]float32, weightElements), zeroZ: make([]float32, hiddenElements), zeroEOS: make([]float32, frames),
		dZ: make([]float32, hiddenElements), directTarget: make([]float32, latentElements), dEOS: make([]float32, frames),
		dDiagLogvar: make([]float32, frames), dDistillLogvar: make([]float32, frames), dLogvars: make([]float32, weightRows),
		xTime: make([]float32, latent), desired: make([]float32, latent),
		flowGradients: newFlowHeadGradients(flow), discardFlowGradients: newFlowHeadGradients(flow),
		flowLMGradients:      newFlowLMTrainingGradients(flowLM),
		flowLMInputGradients: FlowLMTrainingInputGradients{NormalizedLatents: make([]float32, len(batch.NormalizedLatents)), VoiceLatents: make([]float32, len(batch.VoiceLatents))},
		primaryFlowTape:      primaryFlowTape, endpointFlowTape: endpointFlowTape, primaryFlowBackward: primaryFlowBackward, endpointFlowBackward: endpointFlowBackward,
	}
	return w, nil
}

func (w *TrainingWorkspace) compatible(flowLM *FlowLMTrainingCPU, flow *FlowHeadCPU) bool {
	if w == nil || flowLM == nil || flowLM.Transformer == nil || flow == nil || w.flowLM != flowLM || w.flow != flow || w.transformerWidth != flowLM.Transformer.Width || w.transformerHeads != flowLM.Transformer.Heads || w.transformerHeadDim != flowLM.Transformer.HeadDim || w.transformerContext != flowLM.Transformer.Context || w.transformerMaxPeriod != flowLM.Transformer.MaxPeriod || w.flowLMGradients == nil || w.flowLMGradients.Transformer == nil || w.flowGradients == nil || w.discardFlowGradients == nil || w.primaryFlowTape == nil || w.endpointFlowTape == nil || w.primaryFlowBackward == nil || w.endpointFlowBackward == nil || len(flow.Time) != len(w.flowGradients.Time) || len(flow.Blocks) != len(w.flowGradients.Blocks) || !flowLMGradientMatches(w.flowLMGradients, flowLM) {
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

func transformerGradientMatches(g *TransformerGradients, m *TransformerCPU) bool {
	if g == nil || m == nil || len(g.Layers) != len(m.Layers) || len(g.FinalWeight) != len(m.FinalWeight) || len(g.FinalBias) != len(m.FinalBias) {
		return false
	}
	for i := range m.Layers {
		gradient, layer := &g.Layers[i], &m.Layers[i]
		if len(gradient.Norm1Weight) != len(layer.Norm1Weight) || len(gradient.Norm1Bias) != len(layer.Norm1Bias) || len(gradient.Norm2Weight) != len(layer.Norm2Weight) || len(gradient.Norm2Bias) != len(layer.Norm2Bias) || !linearGradientMatches(gradient.InProjection, layer.InProjection) || !linearGradientMatches(gradient.OutProjection, layer.OutProjection) || !linearGradientMatches(gradient.FC1, layer.FC1) || !linearGradientMatches(gradient.FC2, layer.FC2) || len(gradient.LayerScale1) != len(layer.LayerScale1) || len(gradient.LayerScale2) != len(layer.LayerScale2) {
			return false
		}
	}
	return true
}

func flowLMGradientMatches(g *FlowLMTrainingGradients, m *FlowLMTrainingCPU) bool {
	return g != nil && m != nil && len(g.Embedding) == len(m.Embedding) && len(g.BOS) == len(m.BOS) && len(g.BOSBeforeVoice) == len(m.BOSBeforeVoice) && linearGradientMatches(g.SpeakerProjection, m.SpeakerProjection) && linearGradientMatches(g.Input, m.Input) && linearGradientMatches(g.EOS, m.EOS) && transformerGradientMatches(g.Transformer, m.Transformer)
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
	clearFlowLMTrainingGradients(w.flowLMGradients)
	clear(w.flowLMInputGradients.NormalizedLatents)
	clear(w.flowLMInputGradients.VoiceLatents)
	w.primaryFlowTape.reset()
	w.endpointFlowTape.reset()
	w.primaryFlowBackward.reset()
	w.endpointFlowBackward.reset()
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
