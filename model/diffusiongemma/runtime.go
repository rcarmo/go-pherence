package diffusiongemma

import (
	"fmt"
	"math/rand"
)

// BlockDiffusionState keeps block-diffusion decoding state separate from
// autoregressive KV state. DiffusionGemma denoises a fixed-size token canvas
// over multiple steps, then appends accepted canvases to the prompt context.
type BlockDiffusionState struct {
	CanvasLength int  `json:"canvas_length"`
	Step         int  `json:"step"`
	Stopped      bool `json:"stopped"`
}

// ForwardInput describes one denoising forward call. Prompt/cache ownership is
// intentionally abstract at this layer; the concrete model runtime will decide
// how to encode prior canvases and expose cross-attention/prompt-cache state.
type ForwardInput struct {
	PromptIDs []int `json:"prompt_ids,omitempty"`
	Canvas    []int `json:"canvas"`
	Step      int   `json:"step"`
}

// ForwardOutput contains per-canvas-position logits from the denoiser. Logits
// is laid out as [canvas_length][vocab_size].
type ForwardOutput struct {
	Logits [][]float32 `json:"-"`
}

// Denoiser is the narrow interface needed by the block-diffusion sampler. A
// future native implementation should wrap the DiffusionGemma text/vision model
// forward pass behind this interface.
type Denoiser interface {
	Denoise(ForwardInput) (ForwardOutput, error)
}

// CanvasStep records one denoising iteration for diagnostics and future
// streaming hooks.
type CanvasStep struct {
	Step        int     `json:"step"`
	Temperature float64 `json:"temperature"`
	Accepted    int     `json:"accepted"`
	MeanEntropy float64 `json:"mean_entropy"`
	Stopped     bool    `json:"stopped"`
}

// CanvasResult is the output of a single block-diffusion canvas generation.
type CanvasResult struct {
	Canvas []int               `json:"canvas"`
	Steps  []CanvasStep        `json:"steps"`
	State  BlockDiffusionState `json:"state"`
}

// GenerateCanvas runs the reference block-diffusion loop around an abstract
// denoiser. It is model-agnostic scaffold code: correctness fixtures should be
// built against Transformers before using it for real generation.
func GenerateCanvas(denoiser Denoiser, promptIDs []int, cfg DenoisingConfig, canvasLength, vocabSize int, rng *rand.Rand) (CanvasResult, error) {
	if denoiser == nil {
		return CanvasResult{}, fmt.Errorf("nil DiffusionGemma denoiser")
	}
	if canvasLength <= 0 || vocabSize <= 0 {
		return CanvasResult{}, fmt.Errorf("invalid canvas/vocab dimensions canvas=%d vocab=%d", canvasLength, vocabSize)
	}
	if cfg.MaxDenoisingSteps <= 0 {
		cfg.MaxDenoisingSteps = DefaultDenoisingConfig().MaxDenoisingSteps
	}
	if cfg.Sampler.EntropyBound <= 0 {
		cfg.Sampler.EntropyBound = DefaultDenoisingConfig().Sampler.EntropyBound
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(1))
	}
	state := BlockDiffusionState{CanvasLength: canvasLength}
	canvas := make([]int, canvasLength)
	for i := range canvas {
		canvas[i] = rng.Intn(vocabSize)
	}
	stopper := NewStableConfidentStopper(cfg.StabilityThreshold, cfg.ConfidenceThreshold)
	steps := make([]CanvasStep, 0, cfg.MaxDenoisingSteps)
	for step := cfg.MaxDenoisingSteps; step > 0; step-- {
		state.Step = step
		out, err := denoiser.Denoise(ForwardInput{PromptIDs: promptIDs, Canvas: canvas, Step: step})
		if err != nil {
			return CanvasResult{}, err
		}
		if len(out.Logits) < canvasLength {
			return CanvasResult{}, fmt.Errorf("denoiser returned %d canvas logits, want %d", len(out.Logits), canvasLength)
		}
		temperature := LinearTemperature(cfg.TMin, cfg.TMax, cfg.MaxDenoisingSteps, step)
		denoiserCanvas := make([]int, canvasLength)
		entropy := make([]float64, canvasLength)
		var meanEntropy float64
		for i := 0; i < canvasLength; i++ {
			logits := ApplyTemperature(out.Logits[i], temperature)
			denoiserCanvas[i] = Argmax(logits)
			entropy[i] = TokenEntropyFromLogits(logits)
			meanEntropy += entropy[i]
		}
		meanEntropy /= float64(canvasLength)
		accepted := AcceptCanvas(canvas, denoiserCanvas, entropy, cfg.Sampler.EntropyBound)
		stopped := stopper.ShouldStop(denoiserCanvas, entropy)
		steps = append(steps, CanvasStep{Step: step, Temperature: temperature, Accepted: accepted.Accepted, MeanEntropy: meanEntropy, Stopped: stopped})
		canvas = accepted.Canvas
		if stopped {
			state.Stopped = true
			break
		}
		canvas = RenoiseCanvas(canvas, accepted.AcceptedMask, vocabSize, rng)
	}
	return CanvasResult{Canvas: canvas, Steps: steps, State: state}, nil
}
