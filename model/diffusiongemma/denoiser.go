package diffusiongemma

import "fmt"

// TextDenoiser is the future native tensor-backed DiffusionGemma denoiser. It
// currently validates and owns the metadata/weight binding needed by a forward
// pass, but does not yet implement layer math.
type TextDenoiser struct {
	Shape   Shape
	Weights *TextWeights
	Plan    TextForwardPlan
}

func NewTextDenoiser(shape Shape, weights *TextWeights) (*TextDenoiser, error) {
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
	return &TextDenoiser{Shape: shape, Weights: weights, Plan: plan}, nil
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
	return ForwardOutput{}, fmt.Errorf("DiffusionGemma text denoiser forward is not implemented")
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
	}
}
