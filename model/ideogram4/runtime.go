package ideogram4

import "errors"

var ErrRuntimeNotImplemented = errors.New("ideogram4 runtime execution is not implemented")

// TextConditioner will own the native Qwen3-VL text-only forward path that
// returns concatenated hidden states from Config.ActivationLayers.
type TextConditioner interface {
	ConditionText(prompt string, maxTokens int) (TextConditioning, error)
}

// Denoiser will own the conditional/unconditional Ideogram4 DiT velocity model.
type Denoiser interface {
	Denoise(latents Latents, cond TextConditioning, step FlowStep) (Latents, error)
}

// Scheduler will own FlowMatch Euler timestep and latent update logic.
type Scheduler interface {
	Steps(numSteps int) ([]FlowStep, error)
}

// Decoder will own AutoencoderKLFlux2 latent-to-image decode.
type Decoder interface {
	Decode(latents Latents) (Image, error)
}

type TextConditioning struct {
	Tokens   int
	Features []float32
	Dim      int
}

type Latents struct {
	Batch    int
	Tokens   int
	Channels int
	Data     []float32
}

type FlowStep struct {
	Index int
	Sigma float32
	T     float32
}

type Image struct {
	Width  int
	Height int
	RGB    []byte
}

type Pipeline struct {
	Config      Config
	Conditioner TextConditioner
	Conditional Denoiser
	Uncond      Denoiser
	Scheduler   Scheduler
	Decoder     Decoder
}

func NewPipeline(cfg Config) (*Pipeline, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &Pipeline{Config: cfg}, nil
}

func (p *Pipeline) Generate(prompt string, height, width, steps int) (Image, error) {
	if p == nil {
		return Image{}, ErrRuntimeNotImplemented
	}
	if _, err := p.Config.NewLatents(1, height, width); err != nil {
		return Image{}, err
	}
	if _, err := p.Config.BuildSamplingPlan(height, width, steps, 1, nil, DefaultMaxTextTokens, LogitNormalSchedule{}); err != nil {
		return Image{}, err
	}
	return Image{}, ErrRuntimeNotImplemented
}
