package diffusiongemma

import "fmt"

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
	PromptIDs        []int            `json:"prompt_ids,omitempty"`
	Canvas           []int            `json:"canvas"`
	Step             int              `json:"step"`
	SelfConditioning []float32        `json:"-"`
	SCTempInv        float32          `json:"-"` // 1/t for self-conditioning softmax (set by runtime loop)
	SampleDraws      []float64        `json:"-"` // pre-drawn per-position multinomial uniforms for backend/device sampling
	EncoderKV        []EncoderKVLayer `json:"-"`
}

// ForwardOutput contains per-canvas-position logits from the denoiser. Logits
// is laid out as [canvas_length][vocab_size]. SelfConditioning, when present,
// is the hidden-size soft embedding signal to feed into the next denoising step.
type ForwardOutput struct {
	Logits           [][]float32 `json:"-"`
	SelfConditioning []float32   `json:"-"`
	ArgmaxCanvas     []int       `json:"-"` // optional backend/device-computed argmax per canvas position
	SampledCanvas    []int       `json:"-"` // optional backend/device-computed multinomial sample per canvas position
	Entropy          []float64   `json:"-"` // optional backend/device-computed entropy per canvas position
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
func GenerateCanvas(denoiser Denoiser, promptIDs []int, cfg DenoisingConfig, canvasLength, vocabSize int, rng canvasRNG) (CanvasResult, error) {
	return GenerateCanvasWithCallback(denoiser, promptIDs, cfg, canvasLength, vocabSize, rng, nil)
}

// GenerateCanvasWithCallback is GenerateCanvas with an optional per-step callback.
func GenerateCanvasWithCallback(denoiser Denoiser, promptIDs []int, cfg DenoisingConfig, canvasLength, vocabSize int, rng canvasRNG, onStep func(CanvasStep, []int)) (CanvasResult, error) {
	if denoiser == nil {
		return CanvasResult{}, fmt.Errorf("nil DiffusionGemma denoiser")
	}
	if canvasLength <= 0 || vocabSize <= 0 {
		return CanvasResult{}, fmt.Errorf("invalid canvas/vocab dimensions canvas=%d vocab=%d", canvasLength, vocabSize)
	}
	defaults := DefaultDenoisingConfig()
	if cfg.MaxDenoisingSteps <= 0 {
		cfg.MaxDenoisingSteps = defaults.MaxDenoisingSteps
	}
	if cfg.TMin <= 0 {
		cfg.TMin = defaults.TMin
	}
	if cfg.TMax <= 0 {
		cfg.TMax = defaults.TMax
	}
	if cfg.StabilityThreshold < 0 {
		cfg.StabilityThreshold = defaults.StabilityThreshold
	}
	if cfg.ConfidenceThreshold < 0 {
		cfg.ConfidenceThreshold = defaults.ConfidenceThreshold
	}
	if cfg.Sampler.EntropyBound < 0 {
		cfg.Sampler.EntropyBound = defaults.Sampler.EntropyBound
	}
	if rng == nil {
		rng = NewMT19937RNG(0)
	}
	state := BlockDiffusionState{CanvasLength: canvasLength}
	canvas := make([]int, canvasLength)
	for i := range canvas {
		canvas[i] = rng.Intn(vocabSize)
	}
	// llama.cpp keeps two canvases: current_canvas is the renoised input to the
	// next step, while output_tokens stores the argmax canvas from the latest
	// model forward. Return the latter for text generation correctness.
	outputCanvas := append([]int(nil), canvas...)
	stopper := NewStableConfidentStopper(cfg.StabilityThreshold, cfg.ConfidenceThreshold)
	steps := make([]CanvasStep, 0, cfg.MaxDenoisingSteps)
	var selfConditioning []float32
	for step := cfg.MaxDenoisingSteps; step > 0; step-- {
		state.Step = step
		temperature := LinearTemperature(cfg.TMin, cfg.TMax, cfg.MaxDenoisingSteps, step)
		tempInv := float32(1.0 / temperature)
		// Go stores the already-softened self-conditioning embedding, not raw
		// logits. Therefore the embedding returned from this denoise call must be
		// built from this step's logits with this step's temp_inv; it will be fed
		// into the next forward where the SC MLP is gated on.
		sampleDraws := make([]float64, canvasLength)
		renoiseTokens := make([]int, canvasLength)
		for i := 0; i < canvasLength; i++ {
			sampleDraws[i] = rng.Float64()
			renoiseTokens[i] = rng.Intn(vocabSize)
		}
		out, err := denoiser.Denoise(ForwardInput{PromptIDs: promptIDs, Canvas: canvas, Step: step, SelfConditioning: selfConditioning, SCTempInv: tempInv, SampleDraws: sampleDraws})
		if err != nil {
			return CanvasResult{}, err
		}
		if len(out.Logits) < canvasLength && (len(out.ArgmaxCanvas) < canvasLength || len(out.SampledCanvas) < canvasLength || len(out.Entropy) < canvasLength) {
			return CanvasResult{}, fmt.Errorf("denoiser returned logits=%d argmax=%d sampled=%d entropy=%d, want canvas=%d", len(out.Logits), len(out.ArgmaxCanvas), len(out.SampledCanvas), len(out.Entropy), canvasLength)
		}
		denoiserCanvas := make([]int, canvasLength)
		argmaxCanvas := make([]int, canvasLength)
		entropy := make([]float64, canvasLength)
		var meanEntropy float64
		if len(out.ArgmaxCanvas) >= canvasLength && len(out.SampledCanvas) >= canvasLength && len(out.Entropy) >= canvasLength {
			copy(argmaxCanvas, out.ArgmaxCanvas[:canvasLength])
			copy(denoiserCanvas, out.SampledCanvas[:canvasLength])
			copy(entropy, out.Entropy[:canvasLength])
			for _, v := range entropy {
				meanEntropy += v
			}
		} else {
			for i := 0; i < canvasLength; i++ {
				logits := ApplyTemperature(out.Logits[i], temperature)
				argmaxCanvas[i] = Argmax(logits)
				denoiserCanvas[i] = SampleFromLogitsWithDraw(logits, sampleDraws[i])
				entropy[i] = TokenEntropyFromLogits(logits)
				meanEntropy += entropy[i]
			}
		}
		meanEntropy /= float64(canvasLength)
		if len(out.SelfConditioning) > 0 {
			selfConditioning = append(selfConditioning[:0], out.SelfConditioning...)
		} else {
			selfConditioning = nil
		}
		copy(outputCanvas, argmaxCanvas)
		stopped := stopper.ShouldStop(argmaxCanvas, entropy)
		// Always use entropy-bound acceptance + renoise for the next input canvas
		// (same as llama.cpp); do not return this working canvas as generated text.
		accepted := AcceptCanvas(canvas, denoiserCanvas, entropy, cfg.Sampler.EntropyBound)
		canvas = accepted.Canvas
		if !stopped {
			canvas = RenoiseCanvasWithTokens(canvas, accepted.AcceptedMask, renoiseTokens)
		}
		steps = append(steps, CanvasStep{Step: step, Temperature: temperature, Accepted: accepted.Accepted, MeanEntropy: meanEntropy, Stopped: stopped})
		if onStep != nil {
			onStep(steps[len(steps)-1], canvas)
		}
		if stopped {
			state.Stopped = true
			break
		}
	}
	return CanvasResult{Canvas: outputCanvas, Steps: steps, State: state}, nil
}
