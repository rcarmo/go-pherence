package diffusiongemma

import "fmt"

// TextDenoiser is the future native tensor-backed DiffusionGemma denoiser. It
// currently validates and owns the metadata/weight binding needed by a forward
// pass, but does not yet implement layer math.
type TextDenoiser struct {
	Shape      Shape
	Weights    *TextWeights
	Plan       TextForwardPlan
	Ops        ForwardOpPlan
	Buffers    ForwardBufferPlan
	Dispatcher ForwardDispatcher
	EncoderKV  []EncoderKVLayer
}

func NewTextDenoiser(shape Shape, weights *TextWeights) (*TextDenoiser, error) {
	return NewTextDenoiserWithDispatcher(shape, weights, NotImplementedDispatcher{})
}

func NewTextDenoiserWithDispatcher(shape Shape, weights *TextWeights, dispatcher ForwardDispatcher) (*TextDenoiser, error) {
	if weights == nil {
		return nil, fmt.Errorf("nil DiffusionGemma text weights")
	}
	plan := weights.ForwardPlan()
	if !plan.Ready {
		return nil, fmt.Errorf("DiffusionGemma text forward plan incomplete: %v", plan.Missing)
	}
	if shape.CanvasLength <= 0 || shape.VocabSize <= 0 || shape.TextHiddenSize <= 0 || shape.TextLayers <= 0 {
		return nil, fmt.Errorf("invalid DiffusionGemma shape for denoiser")
	}
	if len(plan.Layers) != shape.TextLayers {
		return nil, fmt.Errorf("DiffusionGemma text layer binding count=%d want %d", len(plan.Layers), shape.TextLayers)
	}
	if dispatcher == nil {
		dispatcher = NotImplementedDispatcher{}
	}
	ops := BuildForwardOpPlan(shape, &plan)
	if !ops.Ready {
		return nil, fmt.Errorf("DiffusionGemma text op plan not ready: %s", ops.Reason)
	}
	return &TextDenoiser{Shape: shape, Weights: weights, Plan: plan, Ops: ops, Buffers: BuildForwardBufferPlan(shape), Dispatcher: dispatcher}, nil
}

func (d *TextDenoiser) Denoise(in ForwardInput) (ForwardOutput, error) {
	if d == nil {
		return ForwardOutput{}, fmt.Errorf("nil DiffusionGemma text denoiser")
	}
	if len(in.Canvas) == 0 {
		return ForwardOutput{}, fmt.Errorf("empty DiffusionGemma canvas")
	}
	if d.Shape.VocabSize <= 0 {
		return ForwardOutput{}, fmt.Errorf("invalid DiffusionGemma vocab size %d", d.Shape.VocabSize)
	}
	if d.Dispatcher == nil {
		return ForwardOutput{}, fmt.Errorf("DiffusionGemma text forward dispatcher is not configured")
	}
	// Encode prompt on first call if not already done
	if d.EncoderKV == nil && len(in.PromptIDs) > 0 {
		var encDispatcher CPUDispatcher
		var fp8w *FP8TextWeights
		var expertCache *ExpertLRUCache
		switch disp := d.Dispatcher.(type) {
		case CPUDispatcher:
			encDispatcher = disp
		case GPUDispatcher:
			encDispatcher = disp.cpuFallback()
			fp8w = disp.FP8Weights
			expertCache = disp.ExpertCache
		default:
			encDispatcher = CPUDispatcher{}
		}
		kv, err := encDispatcher.EncodePromptWithFP8(in.PromptIDs, d.Weights, d.Ops, d.Buffers, nil, fp8w, expertCache)
		if err != nil {
			return ForwardOutput{}, fmt.Errorf("DiffusionGemma encoder: %w", err)
		}
		d.EncoderKV = kv
		// Clear encoder experts from GPU cache so decoder has full cache capacity
		if gd, ok := d.Dispatcher.(GPUDispatcher); ok && gd.ExpertCache != nil {
			gd.ExpertCache.ClearAll()
		}
	}
	encoderSeqLen := 0
	if len(d.EncoderKV) > 0 {
		encoderSeqLen = d.EncoderKV[0].SeqLen
	}
	return d.Dispatcher.RunTextForward(ForwardContext{PromptIDs: in.PromptIDs, Canvas: in.Canvas, Step: in.Step, SelfConditioning: in.SelfConditioning, EncoderKV: d.EncoderKV, EncoderSeqLen: encoderSeqLen}, d.Weights, d.Ops, d.Buffers)
}

// ForwardBufferPlan describes the major scratch buffers required by a future
// CPU/SIMD text denoiser forward pass. Sizes are element counts, not bytes.
type ForwardBufferPlan struct {
	CanvasLength int `json:"canvas_length"`
	HiddenSize   int `json:"hidden_size"`
	VocabSize    int `json:"vocab_size"`
	Hidden       int `json:"hidden"`
	Residual     int `json:"residual"`
	Logits       int `json:"logits"`
	Router       int `json:"router"`
	Experts      int `json:"experts"`
	TopKExperts  int `json:"top_k_experts"`
}

func BuildForwardBufferPlan(shape Shape) ForwardBufferPlan {
	canvas := shape.CanvasLength
	if canvas < 0 {
		canvas = 0
	}
	hidden := shape.TextHiddenSize
	if hidden < 0 {
		hidden = 0
	}
	vocab := shape.VocabSize
	if vocab < 0 {
		vocab = 0
	}
	return ForwardBufferPlan{
		CanvasLength: canvas,
		HiddenSize:   hidden,
		VocabSize:    vocab,
		Hidden:       canvas * hidden,
		Residual:     canvas * hidden,
		Logits:       canvas * vocab,
		Router:       canvas * shape.NumExperts,
		Experts:      canvas * shape.TopKExperts * shape.MoEIntermediateSize,
		TopKExperts:  shape.TopKExperts,
	}
}
