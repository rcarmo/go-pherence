package pockettts

import (
	"fmt"
	"math"
)

const (
	ProductionTrainingTargetFrames = 30 * FrameRateNumerator / FrameRateDenominator // 375
	ProductionTrainingVoiceFrames  = 5 * FrameRateNumerator / FrameRateDenominator  // 62, upstream int()
)

// TrainingShape is one native CPU micro-batch. The current exact graph is
// intentionally one row; larger effective batches use gradient accumulation.
type TrainingShape struct {
	MicroBatchRows       int
	TargetFrames         int
	VoiceFrames          int
	TextTokens           int
	GradientAccumulation int
	FlowBatchMultiplier  int
}

// TrainingShapeLimits makes every production admission ceiling explicit.
type TrainingShapeLimits struct {
	MaxTargetFrames  int
	MaxVoiceFrames   int
	MaxTextTokens    int
	MaxSequenceRows  int
	MaxResidentBytes int64
}

// TrainingShapePlan is allocation-free. Parameter/state byte counts are exact
// for the configured topology and normalized LSD defaults. ActivationUpperBytes
// includes four copies of the calculated live tape/scratch set to cover transient
// clones, allocator rounding, and one garbage-collector cycle of dead tapes.
type TrainingShapePlan struct {
	TargetFrames         int
	VoiceFrames          int
	TextTokens           int
	Hidden               int
	Latent               int
	SequenceRows         int
	EffectiveBatchRows   int
	TransformerLayers    int
	ParameterElements    int64
	ParameterBytes       int64
	GradientBytes        int64
	AdamMomentBytes      int64
	EMABytes             int64
	ActivationUpperBytes int64
	FlowScratchBytes     int64
	ResidentUpperBytes   int64
	admission            trainingShapeAdmission
}

type trainingShapeAdmission struct {
	targetFrames, voiceFrames, textTokens                 int
	hidden, latent, heads, transformerLayers, feedForward int
	vocabulary, flowDepth, flowDim                        int
	transformerContext                                    int
	transformerMaxPeriod                                  float64
	layerScale                                            bool
	parameterElements                                     int64
}

// ProductionTrainingShapeLimits returns the pinned upstream 30 s/5 s envelope.
// Text is explicitly bounded at 512 tokens because upstream has no tokenizer
// length cap; sequence includes BOS, voice, text and target rows.
func ProductionTrainingShapeLimits(maxResidentBytes int64) TrainingShapeLimits {
	return TrainingShapeLimits{MaxTargetFrames: ProductionTrainingTargetFrames, MaxVoiceFrames: ProductionTrainingVoiceFrames, MaxTextTokens: 512, MaxSequenceRows: 1 + ProductionTrainingVoiceFrames + 512 + ProductionTrainingTargetFrames, MaxResidentBytes: maxResidentBytes}
}

// PlanTrainingShape validates a production row before any model-sized tensor or
// tape allocation. Native batching is not implemented: MicroBatchRows must be
// one and GradientAccumulation carries the effective batch dimension.
func PlanTrainingShape(cfg Config, shape TrainingShape, limits TrainingShapeLimits) (TrainingShapePlan, error) {
	var plan TrainingShapePlan
	if err := cfg.Validate(); err != nil {
		return plan, err
	}
	if shape.MicroBatchRows != 1 || shape.TargetFrames <= 0 || shape.VoiceFrames < 0 || shape.TextTokens <= 0 || shape.GradientAccumulation <= 0 || shape.FlowBatchMultiplier != 1 {
		return plan, fmt.Errorf("invalid Pocket TTS native training shape")
	}
	if limits.MaxTargetFrames <= 0 || limits.MaxVoiceFrames < 0 || limits.MaxTextTokens <= 0 || limits.MaxSequenceRows <= 0 || limits.MaxResidentBytes <= 0 {
		return plan, fmt.Errorf("invalid Pocket TTS training shape limits")
	}
	if shape.TargetFrames > limits.MaxTargetFrames || shape.VoiceFrames > limits.MaxVoiceFrames || shape.TextTokens > limits.MaxTextTokens {
		return plan, fmt.Errorf("Pocket TTS training row exceeds target/voice/text limits")
	}
	rows, ok := shapeInt64Add(1, int64(shape.VoiceFrames), int64(shape.TextTokens), int64(shape.TargetFrames))
	if !ok || rows > int64(limits.MaxSequenceRows) || rows > int64(math.MaxInt) {
		return plan, fmt.Errorf("Pocket TTS training sequence exceeds limit")
	}
	effective, ok := shapeInt64Mul(int64(shape.MicroBatchRows), int64(shape.GradientAccumulation))
	if !ok || effective > int64(math.MaxInt) {
		return plan, fmt.Errorf("Pocket TTS effective batch overflows")
	}

	parameters, ok := trainingParameterElements(cfg)
	if !ok {
		return plan, fmt.Errorf("Pocket TTS training parameter shape overflows")
	}
	parameterBytes, ok := shapeInt64Mul(parameters, 4)
	if !ok {
		return plan, fmt.Errorf("Pocket TTS training parameter bytes overflow")
	}
	moments, ok := shapeInt64Mul(parameterBytes, 2)
	if !ok {
		return plan, fmt.Errorf("Pocket TTS Adam state bytes overflow")
	}
	activationElements, ok := trainingActivationUpperElements(cfg, rows, int64(shape.TargetFrames), int64(shape.VoiceFrames))
	if !ok {
		return plan, fmt.Errorf("Pocket TTS activation shape overflows")
	}
	activationBytes, ok := shapeInt64Mul(activationElements, 4)
	if !ok {
		return plan, fmt.Errorf("Pocket TTS activation bytes overflow")
	}
	flowGradientElements, ok := trainingFlowGradientElements(cfg)
	if !ok {
		return plan, fmt.Errorf("Pocket TTS flow scratch shape overflows")
	}
	flowScratchBytes, ok := shapeInt64Mul(flowGradientElements, 4)
	if !ok {
		return plan, fmt.Errorf("Pocket TTS flow scratch bytes overflow")
	}
	resident, ok := shapeInt64Add(parameterBytes, parameterBytes, moments, parameterBytes, activationBytes, flowScratchBytes)
	if !ok || resident > limits.MaxResidentBytes {
		return plan, fmt.Errorf("Pocket TTS training resident upper bytes=%d exceed limit=%d", resident, limits.MaxResidentBytes)
	}
	ff := cfg.FlowLM.Transformer.DimFeedforward
	if ff == 0 {
		ff = cfg.FlowLM.Transformer.DModel * cfg.FlowLM.Transformer.HiddenScale
	}
	admission := trainingShapeAdmission{targetFrames: shape.TargetFrames, voiceFrames: shape.VoiceFrames, textTokens: shape.TextTokens, hidden: cfg.FlowLM.Transformer.DModel, latent: cfg.Mimi.InnerDim, heads: cfg.FlowLM.Transformer.NumHeads, transformerLayers: cfg.FlowLM.Transformer.NumLayers, feedForward: ff, vocabulary: cfg.FlowLM.LookupTable.NBins + 1, flowDepth: cfg.FlowLM.Flow.Depth, flowDim: cfg.FlowLM.Flow.Dim, transformerContext: cfg.FlowLM.Transformer.Context, transformerMaxPeriod: cfg.FlowLM.Transformer.MaxPeriod, layerScale: cfg.FlowLM.Transformer.LayerScale != 0, parameterElements: parameters}
	plan = TrainingShapePlan{TargetFrames: shape.TargetFrames, VoiceFrames: shape.VoiceFrames, TextTokens: shape.TextTokens, Hidden: cfg.FlowLM.Transformer.DModel, Latent: cfg.Mimi.InnerDim, SequenceRows: int(rows), EffectiveBatchRows: int(effective), TransformerLayers: cfg.FlowLM.Transformer.NumLayers, ParameterElements: parameters, ParameterBytes: parameterBytes, GradientBytes: parameterBytes, AdamMomentBytes: moments, EMABytes: parameterBytes, ActivationUpperBytes: activationBytes, FlowScratchBytes: flowScratchBytes, ResidentUpperBytes: resident, admission: admission}
	return plan, nil
}

func trainingParameterElements(cfg Config) (int64, bool) {
	h, c, f := int64(cfg.FlowLM.Transformer.DModel), int64(cfg.Mimi.InnerDim), int64(cfg.FlowLM.Flow.Dim)
	layers, depth, vocab := int64(cfg.FlowLM.Transformer.NumLayers), int64(cfg.FlowLM.Flow.Depth), int64(cfg.FlowLM.LookupTable.NBins)
	ff, ok := shapeInt64Mul(h, int64(cfg.FlowLM.Transformer.HiddenScale))
	if cfg.FlowLM.Transformer.DimFeedforward != 0 {
		ff, ok = int64(cfg.FlowLM.Transformer.DimFeedforward), true
	}
	if !ok {
		return 0, false
	}
	vocabWithPad, ok := shapeInt64Add(vocab, 1)
	if !ok {
		return 0, false
	}
	embedding, ok := shapeInt64Mul(vocabWithPad, h)
	if !ok {
		return 0, false
	}
	ch, ok := shapeInt64Mul(c, h)
	if !ok {
		return 0, false
	}
	twoH, ok := shapeInt64Mul(2, h)
	if !ok {
		return 0, false
	}
	globals, ok := shapeInt64Add(embedding, c, h, ch, ch, h, 1, twoH)
	if !ok {
		return 0, false
	}
	hh, ok := shapeInt64Mul(h, h)
	if !ok {
		return 0, false
	}
	hff, ok := shapeInt64Mul(h, ff)
	if !ok {
		return 0, false
	}
	ffh, ok := shapeInt64Mul(ff, h)
	if !ok {
		return 0, false
	}
	qkv, ok := shapeInt64Mul(3, hh)
	if !ok {
		return 0, false
	}
	fourH, ok := shapeInt64Mul(4, h)
	if !ok {
		return 0, false
	}
	layer, ok := shapeInt64Add(fourH, qkv, hh, hff, ffh)
	if !ok {
		return 0, false
	}
	if cfg.FlowLM.Transformer.LayerScale != 0 {
		layer, ok = shapeInt64Add(layer, twoH)
		if !ok {
			return 0, false
		}
	}
	transformer, ok := shapeInt64Mul(layers, layer)
	if !ok {
		return 0, false
	}
	cf, ok := shapeInt64Mul(c, f)
	if !ok {
		return 0, false
	}
	hf, ok := shapeInt64Mul(h, f)
	if !ok {
		return 0, false
	}
	fflow, ok := shapeInt64Mul(f, f)
	if !ok {
		return 0, false
	}
	freqProjection, ok := shapeInt64Mul(256, f)
	if !ok {
		return 0, false
	}
	timeOne, ok := shapeInt64Add(freqProjection, f, fflow, f, f)
	if !ok {
		return 0, false
	}
	timeAll, ok := shapeInt64Mul(2, timeOne)
	if !ok {
		return 0, false
	}
	fiveFFlow, ok := shapeInt64Mul(5, fflow)
	if !ok {
		return 0, false
	}
	sevenF, ok := shapeInt64Mul(7, f)
	if !ok {
		return 0, false
	}
	blockOne, ok := shapeInt64Add(fiveFFlow, sevenF)
	if !ok {
		return 0, false
	}
	blocks, ok := shapeInt64Mul(depth, blockOne)
	if !ok {
		return 0, false
	}
	twoFFlow, ok := shapeInt64Mul(2, fflow)
	if !ok {
		return 0, false
	}
	twoF, ok := shapeInt64Mul(2, f)
	if !ok {
		return 0, false
	}
	finalMod, ok := shapeInt64Add(twoFFlow, twoF)
	if !ok {
		return 0, false
	}
	finalLinear, ok := shapeInt64Add(cf, c)
	if !ok {
		return 0, false
	}
	flowGlobals, ok := shapeInt64Add(cf, f, hf, f, timeAll, blocks, finalMod, finalLinear)
	if !ok {
		return 0, false
	}
	const weighting = int64(2*32 + 32 + 32*32 + 32 + 32*32 + 32 + 32 + 1)
	return shapeInt64Add(globals, transformer, flowGlobals, weighting)
}

func trainingFlowGradientElements(cfg Config) (int64, bool) {
	c, h, f := int64(cfg.Mimi.InnerDim), int64(cfg.FlowLM.Transformer.DModel), int64(cfg.FlowLM.Flow.Dim)
	depth := int64(cfg.FlowLM.Flow.Depth)
	cf, ok := shapeInt64Mul(c, f)
	if !ok {
		return 0, false
	}
	hf, ok := shapeInt64Mul(h, f)
	if !ok {
		return 0, false
	}
	ff, ok := shapeInt64Mul(f, f)
	if !ok {
		return 0, false
	}
	freq, ok := shapeInt64Mul(256, f)
	if !ok {
		return 0, false
	}
	timeOne, ok := shapeInt64Add(freq, f, ff, f, f)
	if !ok {
		return 0, false
	}
	times, ok := shapeInt64Mul(2, timeOne)
	if !ok {
		return 0, false
	}
	threeFF, ok := shapeInt64Mul(3, ff)
	if !ok {
		return 0, false
	}
	threeF, ok := shapeInt64Mul(3, f)
	if !ok {
		return 0, false
	}
	block, ok := shapeInt64Add(f, f, ff, f, ff, f, threeFF, threeF)
	if !ok {
		return 0, false
	}
	blocks, ok := shapeInt64Mul(depth, block)
	if !ok {
		return 0, false
	}
	twoFF, ok := shapeInt64Mul(2, ff)
	if !ok {
		return 0, false
	}
	twoF, ok := shapeInt64Mul(2, f)
	if !ok {
		return 0, false
	}
	final, ok := shapeInt64Add(cf, c, twoFF, twoF)
	if !ok {
		return 0, false
	}
	one, ok := shapeInt64Add(cf, f, hf, f, times, blocks, final)
	if !ok {
		return 0, false
	}
	return one, true
}

func trainingActivationUpperElements(cfg Config, rows, target, voice int64) (int64, bool) {
	h, c, f := int64(cfg.FlowLM.Transformer.DModel), int64(cfg.Mimi.InnerDim), int64(cfg.FlowLM.Flow.Dim)
	heads, layers, depth := int64(cfg.FlowLM.Transformer.NumHeads), int64(cfg.FlowLM.Transformer.NumLayers), int64(cfg.FlowLM.Flow.Depth)
	ff, ok := shapeInt64Mul(h, int64(cfg.FlowLM.Transformer.HiddenScale))
	if cfg.FlowLM.Transformer.DimFeedforward != 0 {
		ff, ok = int64(cfg.FlowLM.Transformer.DimFeedforward), true
	}
	if !ok {
		return 0, false
	}
	rh, ok := shapeInt64Mul(rows, h)
	if !ok {
		return 0, false
	}
	rff, ok := shapeInt64Mul(rows, ff)
	if !ok {
		return 0, false
	}
	rr, ok := shapeInt64Mul(rows, rows)
	if !ok {
		return 0, false
	}
	headRR, ok := shapeInt64Mul(heads, rr)
	if !ok {
		return 0, false
	}
	persistentH, ok := shapeInt64Mul(13, rh)
	if !ok {
		return 0, false
	}
	persistentFF, ok := shapeInt64Mul(2, rff)
	if !ok {
		return 0, false
	}
	twoRows, ok := shapeInt64Mul(2, rows)
	if !ok {
		return 0, false
	}
	perLayer, ok := shapeInt64Add(persistentH, persistentFF, headRR, twoRows)
	if !ok {
		return 0, false
	}
	persistent, ok := shapeInt64Mul(layers, perLayer)
	if !ok {
		return 0, false
	}
	threeRH, ok := shapeInt64Mul(3, rh)
	if !ok {
		return 0, false
	}
	finalTape, ok := shapeInt64Add(threeRH, rows)
	if !ok {
		return 0, false
	}
	targetC, ok := shapeInt64Mul(target, c)
	if !ok {
		return 0, false
	}
	targetH, ok := shapeInt64Mul(target, h)
	if !ok {
		return 0, false
	}
	voiceH, ok := shapeInt64Mul(voice, h)
	if !ok {
		return 0, false
	}
	twoRH, ok := shapeInt64Mul(2, rh)
	if !ok {
		return 0, false
	}
	flowLM, ok := shapeInt64Add(twoRH, targetC, targetH, target, voiceH)
	if !ok {
		return 0, false
	}
	headRows, ok := shapeInt64Mul(heads, rows)
	if !ok {
		return 0, false
	}
	eighteenRH, ok := shapeInt64Mul(18, rh)
	if !ok {
		return 0, false
	}
	threeRFF, ok := shapeInt64Mul(3, rff)
	if !ok {
		return 0, false
	}
	threeH, ok := shapeInt64Mul(3, h)
	if !ok {
		return 0, false
	}
	backward, ok := shapeInt64Add(eighteenRH, threeRFF, headRows, threeH, ff)
	if !ok {
		return 0, false
	}
	twoTargetH, ok := shapeInt64Mul(2, targetH)
	if !ok {
		return 0, false
	}
	tenTarget, ok := shapeInt64Mul(10, target)
	if !ok {
		return 0, false
	}
	twoC, ok := shapeInt64Mul(2, c)
	if !ok {
		return 0, false
	}
	workspace, ok := shapeInt64Add(twoTargetH, targetC, tenTarget, twoC)
	if !ok {
		return 0, false
	}
	depthTerm, ok := shapeInt64Mul(36, depth)
	if !ok {
		return 0, false
	}
	flowFactor, ok := shapeInt64Add(54, depthTerm)
	if !ok {
		return 0, false
	}
	flowWidth, ok := shapeInt64Mul(flowFactor, f)
	if !ok {
		return 0, false
	}
	fiveC, ok := shapeInt64Mul(5, c)
	if !ok {
		return 0, false
	}
	flowRow, ok := shapeInt64Add(fiveC, threeH, flowWidth, 1536)
	if !ok {
		return 0, false
	}
	flowRow, ok = shapeInt64Mul(flowRow, 4)
	if !ok {
		return 0, false
	}
	total, ok := shapeInt64Add(persistent, finalTape, flowLM, backward, workspace, flowRow)
	if !ok {
		return 0, false
	}
	return shapeInt64Mul(total, 4)
}

func shapeInt64Add(values ...int64) (int64, bool) {
	var total int64
	for _, value := range values {
		if value < 0 || total > math.MaxInt64-value {
			return 0, false
		}
		total += value
	}
	return total, true
}
func shapeInt64Mul(a, b int64) (int64, bool) {
	if a < 0 || b < 0 || (b != 0 && a > math.MaxInt64/b) {
		return 0, false
	}
	return a * b, true
}
