package diffusiongemma

import (
	"fmt"
	"math/rand"
)

// InferenceOptions controls text-only block-diffusion generation. The scaffold
// currently supports token IDs only; processor/chat-template integration stays
// outside the model package.
type InferenceOptions struct {
	MaxNewTokens int                                                                 `json:"max_new_tokens"`
	CanvasLength int                                                                 `json:"canvas_length"`
	Denoising    *DenoisingConfig                                                    `json:"denoising,omitempty"`
	Seed         int64                                                               `json:"seed,omitempty"`
	StepCallback func(generatedTokenIndex int, snapshot DiffusionStepSnapshot) error `json:"-"`
}

// InferenceResult contains generated token IDs and per-canvas diagnostics.
type InferenceResult struct {
	Generated []int          `json:"generated"`
	Canvases  []CanvasResult `json:"canvases"`
}

// Engine ties DiffusionGemma metadata/weights to a denoiser implementation.
// The native denoiser is intentionally injected: the block-diffusion control
// loop is usable before the full tensor-backed forward path is complete.
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
		return InferenceResult{}, fmt.Errorf("DiffusionGemma denoiser is not implemented")
	}
	shape := e.Model.Shape
	canvasLength := opts.CanvasLength
	if canvasLength <= 0 {
		canvasLength = shape.CanvasLength
	}
	if canvasLength <= 0 {
		return InferenceResult{}, fmt.Errorf("invalid DiffusionGemma canvas length %d", canvasLength)
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
	seed := opts.Seed
	if seed == 0 {
		seed = 1
	}
	rng := rand.New(rand.NewSource(seed))
	generated := make([]int, 0, maxNew)
	canvases := make([]CanvasResult, 0, (maxNew+canvasLength-1)/canvasLength)
	context := append([]int(nil), promptIDs...)
	for len(generated) < maxNew {
		canvasDenoiseCfg := denoiseCfg
		if opts.StepCallback != nil {
			generatedIndex := len(generated)
			previous := canvasDenoiseCfg.StepCallback
			canvasDenoiseCfg.StepCallback = func(snapshot DiffusionStepSnapshot) error {
				if previous != nil {
					if err := previous(snapshot); err != nil {
						return err
					}
				}
				return opts.StepCallback(generatedIndex, snapshot)
			}
		}
		canvas, err := GenerateCanvas(e.Denoiser, context, canvasDenoiseCfg, canvasLength, vocabSize, rng)
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
		if take == 0 {
			break
		}
	}
	return InferenceResult{Generated: generated, Canvases: canvases}, nil
}
