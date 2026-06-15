package diffusiongemma

import "fmt"

// InferenceOptions controls text-only block-diffusion generation. The model
// package operates on token IDs; processor/chat-template integration is handled
// by callers such as the runner/server.
// ProgressEvent is emitted during generation for streaming.
type ProgressEvent struct {
	Type        string      `json:"type"` // "step" or "canvas"
	CanvasIndex int         `json:"canvas_index"`
	Step        *CanvasStep `json:"step,omitempty"`
	TokensSoFar int         `json:"tokens_so_far"`
	PartialText string      `json:"partial_text,omitempty"` // decoded partial output
}

type InferenceOptions struct {
	MaxNewTokens int                 `json:"max_new_tokens"`
	CanvasLength int                 `json:"canvas_length"`
	Denoising    *DenoisingConfig    `json:"denoising,omitempty"`
	Seed         int64               `json:"seed,omitempty"`
	OnProgress   func(ProgressEvent) `json:"-"` // streaming callback
}

// InferenceResult contains generated token IDs and per-canvas diagnostics.
type InferenceResult struct {
	Generated []int          `json:"generated"`
	Canvases  []CanvasResult `json:"canvases"`
}

// Engine ties DiffusionGemma metadata/weights to a configured denoiser
// implementation. The denoiser is injected so callers can choose CPU, GPU, GGUF,
// FP8, or mock dispatch paths explicitly.
type Engine struct {
	Model    *Model
	Weights  *TextWeights
	Denoiser Denoiser
}

func NewEngine(model *Model, denoiser Denoiser) (*Engine, error) {
	if model == nil {
		return nil, fmt.Errorf("nil DiffusionGemma model")
	}
	return &Engine{Model: model, Denoiser: denoiser}, nil
}

func NewEngineWithTextWeights(model *Model, weights *TextWeights, denoiser Denoiser) (*Engine, error) {
	eng, err := NewEngine(model, denoiser)
	if err != nil {
		return nil, err
	}
	eng.Weights = weights
	return eng, nil
}

func (e *Engine) GenerateTokenIDs(promptIDs []int, opts InferenceOptions) (InferenceResult, error) {
	if e == nil || e.Model == nil {
		return InferenceResult{}, fmt.Errorf("nil DiffusionGemma inference engine")
	}
	if e.Denoiser == nil {
		return InferenceResult{}, fmt.Errorf("DiffusionGemma denoiser is not configured")
	}
	// Reset encoder KV cache so each request encodes its own prompt.
	if td, ok := e.Denoiser.(*TextDenoiser); ok {
		td.EncoderKV = nil
		td.EncoderPromptIDs = nil
		td.EncoderPromptLen = 0
	}
	shape := e.Model.Shape
	canvasLength := opts.CanvasLength
	if canvasLength <= 0 {
		canvasLength = shape.CanvasLength
	}
	if canvasLength <= 0 {
		return InferenceResult{}, fmt.Errorf("invalid DiffusionGemma canvas length %d", canvasLength)
	}
	if shape.CanvasLength > 0 && canvasLength > shape.CanvasLength {
		return InferenceResult{}, fmt.Errorf("DiffusionGemma canvas length %d exceeds model canvas_length %d", canvasLength, shape.CanvasLength)
	}
	maxNew := opts.MaxNewTokens
	if maxNew <= 0 {
		if e.Model.GenerationDefaults != nil && e.Model.GenerationDefaults.MaxNewTokens > 0 {
			maxNew = e.Model.GenerationDefaults.MaxNewTokens
		} else {
			maxNew = canvasLength
		}
	}
	vocabSize := shape.VocabSize
	if vocabSize <= 0 {
		return InferenceResult{}, fmt.Errorf("invalid DiffusionGemma vocab size %d", vocabSize)
	}
	denoiseCfg := e.Model.Denoising
	if opts.Denoising != nil {
		denoiseCfg = *opts.Denoising
	}
	rng := NewMT19937RNG(opts.Seed)
	generated := make([]int, 0, maxNew)
	canvases := make([]CanvasResult, 0, (maxNew+canvasLength-1)/canvasLength)
	context := append([]int(nil), promptIDs...)
	for len(generated) < maxNew {
		canvasIdx := len(canvases)
		var stepCb func(CanvasStep, []int)
		if opts.OnProgress != nil {
			stepCb = func(cs CanvasStep, currentCanvas []int) {
				opts.OnProgress(ProgressEvent{
					Type:        "step",
					CanvasIndex: canvasIdx,
					Step:        &cs,
					TokensSoFar: len(generated),
				})
			}
		}
		canvas, err := GenerateCanvasWithCallback(e.Denoiser, context, denoiseCfg, canvasLength, vocabSize, rng, stepCb)
		if err != nil {
			return InferenceResult{}, err
		}
		remaining := maxNew - len(generated)
		take := len(canvas.Canvas)
		if take > remaining {
			take = remaining
		}
		generated = append(generated, canvas.Canvas[:take]...)
		context = append(context, canvas.Canvas[:take]...)
		canvases = append(canvases, canvas)
		if opts.OnProgress != nil {
			opts.OnProgress(ProgressEvent{
				Type:        "canvas",
				CanvasIndex: canvasIdx,
				TokensSoFar: len(generated),
			})
		}
		if take == 0 {
			break
		}
	}
	return InferenceResult{Generated: generated, Canvases: canvases}, nil
}
