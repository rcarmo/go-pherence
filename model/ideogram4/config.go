// Package ideogram4 contains native metadata and runtime scaffolding for
// Ideogram 4 text-to-image checkpoints. Full image generation is intentionally
// staged: loader/config owns Diffusers config parsing, while this package owns
// architecture-level validation and future Go/SIMD runtime contracts.
package ideogram4

import (
	"fmt"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

// Config is the architecture-level subset needed before implementing Qwen3-VL
// conditioning, the Ideogram4 DiT denoiser, FlowMatch sampling, and
// AutoencoderKLFlux2 decoding.
type Config struct {
	Pipeline                 string
	Transformer              string
	UnconditionalTransformer string
	TextEncoder              string
	Tokenizer                string
	Scheduler                string
	VAE                      string
	EmbDim                   int
	NumLayers                int
	NumHeads                 int
	HeadDim                  int
	IntermediateSize         int
	AdaLNDim                 int
	InChannels               int
	LLMFeaturesDim           int
	MRoPESection             []int
	ActivationLayers         []int
	TextHidden               int
	TextLayers               int
	VocabSize                int
	PatchSize                int
	AEScaleFactor            int
	VAELatentChannels        int
}

func FromLoaderConfig(cfg loaderconfig.Ideogram4Config) (Config, error) {
	s := loaderconfig.SummarizeIdeogram4Config(cfg)
	out := Config{
		Pipeline:                 s.Pipeline,
		Transformer:              s.Transformer,
		UnconditionalTransformer: s.UnconditionalTransformer,
		TextEncoder:              s.TextEncoder,
		Tokenizer:                s.Tokenizer,
		Scheduler:                s.Scheduler,
		VAE:                      s.VAE,
		EmbDim:                   s.EmbDim,
		NumLayers:                s.Layers,
		NumHeads:                 s.Heads,
		HeadDim:                  s.HeadDim,
		IntermediateSize:         s.IntermediateSize,
		AdaLNDim:                 s.AdaLNDim,
		InChannels:               s.InChannels,
		LLMFeaturesDim:           s.LLMFeaturesDim,
		MRoPESection:             append([]int(nil), s.MRoPESection...),
		ActivationLayers:         append([]int(nil), s.ActivationLayers...),
		TextHidden:               s.TextHidden,
		TextLayers:               s.TextLayers,
		VocabSize:                s.VocabSize,
		PatchSize:                2,
		AEScaleFactor:            8,
		VAELatentChannels:        cfg.VAE.LatentChannels,
	}
	return out, out.Validate()
}

func (c Config) Validate() error {
	if c.Transformer == "" || c.UnconditionalTransformer == "" || c.TextEncoder == "" || c.Tokenizer == "" || c.Scheduler == "" || c.VAE == "" {
		return fmt.Errorf("invalid Ideogram4 config: missing component name")
	}
	if c.EmbDim <= 0 || c.NumLayers <= 0 || c.NumHeads <= 0 || c.HeadDim <= 0 || c.IntermediateSize <= 0 || c.InChannels <= 0 || c.LLMFeaturesDim <= 0 {
		return fmt.Errorf("invalid Ideogram4 transformer dims: %+v", c)
	}
	if c.EmbDim%c.NumHeads != 0 || c.HeadDim != c.EmbDim/c.NumHeads {
		return fmt.Errorf("invalid Ideogram4 head dims: emb=%d heads=%d head_dim=%d", c.EmbDim, c.NumHeads, c.HeadDim)
	}
	if c.TextHidden <= 0 || c.TextLayers <= 0 || c.VocabSize <= 0 || len(c.ActivationLayers) == 0 {
		return fmt.Errorf("invalid Ideogram4 text encoder dims: %+v", c)
	}
	if want := c.TextHidden * len(c.ActivationLayers); c.LLMFeaturesDim != want {
		return fmt.Errorf("invalid Ideogram4 llm features: got=%d want=%d", c.LLMFeaturesDim, want)
	}
	if c.PatchSize <= 0 || c.AEScaleFactor <= 0 || c.VAELatentChannels <= 0 {
		return fmt.Errorf("invalid Ideogram4 latent/patch dims: patch=%d ae_scale=%d latent_channels=%d", c.PatchSize, c.AEScaleFactor, c.VAELatentChannels)
	}
	return nil
}

func (c Config) ImagePatchMultiple() int {
	if c.PatchSize <= 0 || c.AEScaleFactor <= 0 {
		return 0
	}
	return c.PatchSize * c.AEScaleFactor
}

func (c Config) LatentGrid(height, width int) (gridH, gridW int, err error) {
	if err := c.Validate(); err != nil {
		return 0, 0, err
	}
	patch := c.ImagePatchMultiple()
	if height <= 0 || width <= 0 || height%patch != 0 || width%patch != 0 {
		return 0, 0, fmt.Errorf("Ideogram4 height/width must be positive and divisible by %d: %dx%d", patch, height, width)
	}
	return height / patch, width / patch, nil
}

func (c Config) LatentTokenCount(height, width int) (int, error) {
	gridH, gridW, err := c.LatentGrid(height, width)
	if err != nil {
		return 0, err
	}
	return gridH * gridW, nil
}
