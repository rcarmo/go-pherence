package main

import "fmt"

// HTTP admission limits, not model/kernel limits. Keep generation bounded even
// when tokenizer/template/denoising defaults come from a local checkpoint.
const (
	maxRequestBytes        = 1 << 20
	maxPromptTokens        = 8192
	maxGeneratedTokens     = 4096
	maxCanvasTokens        = 256
	maxDenoisingSteps      = 256
	maxCanvasPositionSteps = 1 << 20
)

func (s *server) validateRequest(req completionRequest, prompt []int) error {
	if s.model == nil || s.engine == nil {
		return fmt.Errorf("inference unavailable")
	}
	if len(prompt) == 0 || len(prompt) > maxPromptTokens {
		return fmt.Errorf("prompt token count outside 1..%d", maxPromptTokens)
	}
	for _, id := range prompt {
		if id < 0 || id >= s.model.Shape.VocabSize {
			return fmt.Errorf("invalid prompt token id %d", id)
		}
	}
	for _, v := range []int{req.MaxTokens, req.MaxNewTokens, req.CanvasLength, req.DenoiseSteps, req.DiffusionSteps} {
		if v < 0 {
			return fmt.Errorf("negative generation control")
		}
	}
	if req.MaxTokens > maxGeneratedTokens || req.MaxNewTokens > maxGeneratedTokens || req.DiffusionSteps > maxDenoisingSteps {
		return fmt.Errorf("generation control exceeds HTTP limit")
	}
	opts := s.options(req, nil)
	canvas := opts.CanvasLength
	if canvas <= 0 {
		canvas = s.model.Shape.CanvasLength
	}
	if canvas <= 0 || canvas > maxCanvasTokens || canvas > s.model.Shape.CanvasLength {
		return fmt.Errorf("canvas exceeds HTTP/model limit")
	}
	tokens := opts.MaxNewTokens
	if tokens <= 0 && s.model.GenerationDefaults != nil {
		tokens = s.model.GenerationDefaults.MaxNewTokens
	}
	if tokens <= 0 {
		tokens = canvas
	}
	if tokens > maxGeneratedTokens {
		return fmt.Errorf("generated token limit is %d", maxGeneratedTokens)
	}
	cfg := s.model.Denoising
	if opts.Denoising != nil {
		cfg = *opts.Denoising
	}
	steps := cfg.MaxDenoisingSteps
	if steps <= 0 {
		steps = 1
	} // matches GenerateCanvas
	if req.DenoiseSteps > maxDenoisingSteps || steps > maxDenoisingSteps {
		return fmt.Errorf("denoising step limit is %d", maxDenoisingSteps)
	}
	// Every partial block still evaluates a full canvas. Values above are bounded
	// before multiplication, including nested denoising and checkpoint defaults.
	blocks := (tokens-1)/canvas + 1
	if blocks*canvas*steps > maxCanvasPositionSteps {
		return fmt.Errorf("canvas-position work budget exceeded")
	}
	return nil
}
