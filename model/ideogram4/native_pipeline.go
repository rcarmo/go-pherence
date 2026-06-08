package ideogram4

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

// weightSource unifies single-file and sharded safetensors access.
type weightSource interface {
	CombinedTensorSource
	Names() []string
}

// openWeights opens a component directory's safetensors, preferring a shard
// index when present, otherwise a single .safetensors file.
func openWeights(dir string) (weightSource, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var single string
	for _, e := range entries {
		name := e.Name()
		if filepath.Ext(name) == ".json" && len(name) > len(".index.json") && name[len(name)-len(".index.json"):] == ".index.json" {
			return safetensors.OpenSharded(filepath.Join(dir, name))
		}
		if filepath.Ext(name) == ".safetensors" {
			single = filepath.Join(dir, name)
		}
	}
	if single == "" {
		return nil, fmt.Errorf("ideogram4: no safetensors found in %s", dir)
	}
	return safetensors.Open(single)
}

// NativePipeline is a fully-loaded native Ideogram4 text-to-image pipeline.
type NativePipeline struct {
	Config      Config
	Tokenizer   *tokenizer.Tokenizer
	Conditioner *QwenVLConditioner
	Cond        *DiTModel
	Uncond      *DiTModel
	VAE         *VAEDecoder
}

// LoadNativePipeline assembles every native component from an Ideogram4
// Diffusers directory: tokenizer, Qwen3-VL conditioner, conditional and
// unconditional FP8 DiT transformers, and the AutoencoderKLFlux2 decoder.
func LoadNativePipeline(modelDir string) (*NativePipeline, error) {
	raw, err := loaderconfig.ReadIdeogram4Config(modelDir)
	if err != nil {
		return nil, err
	}
	cfg, err := FromLoaderConfig(raw)
	if err != nil {
		return nil, err
	}
	tok, err := LoadTokenizer(modelDir)
	if err != nil {
		return nil, err
	}

	textSrc, err := openWeights(filepath.Join(modelDir, "text_encoder"))
	if err != nil {
		return nil, fmt.Errorf("ideogram4 text_encoder weights: %w", err)
	}
	conditioner, err := NewQwenVLConditioner(textSrc, cfg, "language_model")
	if err != nil {
		return nil, err
	}

	condSrc, err := openWeights(filepath.Join(modelDir, "transformer"))
	if err != nil {
		return nil, fmt.Errorf("ideogram4 transformer weights: %w", err)
	}
	condDiT, err := LoadDiTModel(cfg, condSrc)
	if err != nil {
		return nil, err
	}

	uncondSrc, err := openWeights(filepath.Join(modelDir, "unconditional_transformer"))
	if err != nil {
		return nil, fmt.Errorf("ideogram4 unconditional_transformer weights: %w", err)
	}
	uncondDiT, err := LoadDiTModel(cfg, uncondSrc)
	if err != nil {
		return nil, err
	}

	vaeSrc, err := openWeights(filepath.Join(modelDir, "vae"))
	if err != nil {
		return nil, fmt.Errorf("ideogram4 vae weights: %w", err)
	}
	vae, err := NewVAEDecoder(vaeSrc, VAEDecoderOptions{
		BlockOutChannels: raw.VAE.BlockOutChannel,
		LayersPerBlock:   raw.VAE.LayersPerBlock,
		LatentChannels:   raw.VAE.LatentChannels,
		NormNumGroups:    raw.VAE.NormNumGroups,
		ScalingFactor:    float32(raw.VAE.ScalingFactor),
		ShiftFactor:      float32(raw.VAE.ShiftFactor),
		UsePostQuantConv: raw.VAE.UsePostQuant,
		MidAddAttention:  raw.VAE.MidAddAttention,
	})
	if err != nil {
		return nil, err
	}

	return &NativePipeline{
		Config:      cfg,
		Tokenizer:   tok,
		Conditioner: conditioner,
		Cond:        condDiT,
		Uncond:      uncondDiT,
		VAE:         vae,
	}, nil
}

// GenerateOptions controls a native generation run.
type GenerateOptions struct {
	Height        int
	Width         int
	Steps         int
	GuidanceScale float32
	MaxTextTokens int
	InitLatents   []float32 // optional [imageTokens, in_channels]; random if nil
}

// Generate runs the full native text-to-image path: tokenize, Qwen3-VL
// conditioning, FlowMatch denoise loop with asymmetric CFG, unpatchify, and VAE
// decode. InitLatents must be supplied (deterministic) since this package does
// not own an RNG policy.
func (p *NativePipeline) Generate(prompt string, opt GenerateOptions) (Image, error) {
	traceTiming := os.Getenv("GO_PHERENCE_IDEOGRAM4_TIMING") == "1"
	t0 := time.Now()
	mark := func(name string, since time.Time) time.Time {
		if traceTiming {
			fmt.Fprintf(os.Stderr, "timing %s=%s total=%s\n", name, time.Since(since), time.Since(t0))
		}
		return time.Now()
	}
	if p == nil {
		return Image{}, ErrRuntimeNotImplemented
	}
	cfg := p.Config
	gridH, gridW, err := cfg.LatentGrid(opt.Height, opt.Width)
	if err != nil {
		return Image{}, err
	}
	imgTokens := gridH * gridW
	if opt.Steps <= 0 {
		return Image{}, fmt.Errorf("ideogram4 generate: steps=%d", opt.Steps)
	}
	if len(opt.InitLatents) != imgTokens*cfg.InChannels {
		return Image{}, fmt.Errorf("ideogram4 generate: init latents=%d want %d*%d", len(opt.InitLatents), imgTokens, cfg.InChannels)
	}

	maxTok := opt.MaxTextTokens
	if maxTok <= 0 {
		maxTok = DefaultMaxTextTokens
	}
	pt, err := TokenizeChatPrompt(p.Tokenizer, prompt, maxTok)
	if err != nil {
		return Image{}, err
	}
	phaseStart := time.Now()
	textFeatures, err := p.Conditioner.Condition(pt.IDs)
	if err != nil {
		return Image{}, err
	}
	phaseStart = mark("qwen_condition", phaseStart)
	if gpuReleaseAfterPhase() {
		// Qwen uses temporary FP8 wrappers today, but this keeps the phase boundary
		// explicit as more text-encoder residency is added.
		p.Cond.ReleaseGPU()
		p.Uncond.ReleaseGPU()
	}

	plan, err := cfg.BuildSamplingPlan(opt.Height, opt.Width, opt.Steps, opt.GuidanceScale, nil, maxTok, LogitNormalSchedule{})
	if err != nil {
		return Image{}, err
	}
	sched, err := NewFlowMatchScheduler(cfg, opt.Height, opt.Width, LogitNormalSchedule{})
	if err != nil {
		return Image{}, err
	}
	latents, err := DenoiseLoop(p.Cond, p.Uncond, sched, plan, opt.InitLatents, gridH, gridW, textFeatures)
	if err != nil {
		return Image{}, err
	}
	phaseStart = mark("denoise", phaseStart)
	if gpuReleaseAfterPhase() {
		p.Cond.ReleaseGPU()
		p.Uncond.ReleaseGPU()
	}

	patch := cfg.PatchSize
	if err := DenormalizeLatents(latents, cfg.InChannels); err != nil {
		return Image{}, err
	}
	fmap, err := UnpatchifyLatents(latents, gridH, gridW, cfg.InChannels, cfg.VAELatentChannels, patch, patch)
	if err != nil {
		return Image{}, err
	}
	phaseStart = mark("latent_postprocess", phaseStart)
	img, err := p.VAE.Decode(fmap)
	if err != nil {
		return Image{}, err
	}
	mark("vae_decode", phaseStart)
	return img, nil
}
