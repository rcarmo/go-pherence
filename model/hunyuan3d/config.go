// Package hunyuan3d contains metadata and validation helpers for Tencent
// Hunyuan3D shape-generation checkpoints. It does not implement runtime
// inference yet; loader/config owns YAML parsing and tensor inventory.
package hunyuan3d

import (
	"fmt"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

// ShapeConfig is the architecture-level subset needed before implementing the
// conditioner, DiT denoiser, scheduler loop, and VAE decoder.
type ShapeConfig struct {
	DenoiserTarget    string
	VAETarget         string
	ConditionerTarget string
	SchedulerTarget   string
	InChannels        int
	ContextInDim      int
	HiddenSize        int
	NumHeads          int
	HeadDim           int
	Depth             int
	DepthSingleBlocks int
	VAELatents        int
	VAEEmbedDim       int
	VAEWidth          int
	VAEHeads          int
	VAEHeadDim        int
	ConditionerType   string
	SchedulerSteps    int
	GuidanceEmbed     bool
}

func FromLoaderConfig(cfg loaderconfig.Hunyuan3DConfig) (ShapeConfig, error) {
	s := cfg.Summary()
	out := ShapeConfig{
		DenoiserTarget:    s.DenoiserTarget,
		VAETarget:         s.VAETarget,
		ConditionerTarget: s.ConditionerTarget,
		SchedulerTarget:   s.SchedulerTarget,
		InChannels:        s.InChannels,
		ContextInDim:      s.ContextInDim,
		HiddenSize:        s.HiddenSize,
		NumHeads:          s.NumHeads,
		Depth:             s.Depth,
		DepthSingleBlocks: s.DepthSingleBlocks,
		VAELatents:        s.VAELatents,
		VAEEmbedDim:       s.VAEEmbedDim,
		VAEWidth:          s.VAEWidth,
		VAEHeads:          s.VAEHeads,
		ConditionerType:   s.ConditionerType,
		SchedulerSteps:    s.SchedulerSteps,
		GuidanceEmbed:     cfg.Model.Params.GuidanceEmbed,
	}
	if out.NumHeads > 0 {
		out.HeadDim = out.HiddenSize / out.NumHeads
	}
	if out.VAEHeads > 0 {
		out.VAEHeadDim = out.VAEWidth / out.VAEHeads
	}
	return out, out.Validate()
}

func (c ShapeConfig) Validate() error {
	if c.DenoiserTarget == "" || c.VAETarget == "" || c.ConditionerTarget == "" || c.SchedulerTarget == "" {
		return fmt.Errorf("invalid Hunyuan3D shape config: missing component target")
	}
	if c.InChannels <= 0 || c.ContextInDim <= 0 || c.HiddenSize <= 0 || c.NumHeads <= 0 || c.Depth < 0 || c.DepthSingleBlocks < 0 {
		return fmt.Errorf("invalid Hunyuan3D denoiser dims: %+v", c)
	}
	if c.HiddenSize%c.NumHeads != 0 || c.HeadDim != c.HiddenSize/c.NumHeads {
		return fmt.Errorf("invalid Hunyuan3D denoiser head dims: hidden=%d heads=%d head_dim=%d", c.HiddenSize, c.NumHeads, c.HeadDim)
	}
	if c.VAELatents <= 0 || c.VAEEmbedDim <= 0 || c.VAEWidth <= 0 || c.VAEHeads <= 0 {
		return fmt.Errorf("invalid Hunyuan3D VAE dims: %+v", c)
	}
	if c.VAEWidth%c.VAEHeads != 0 || c.VAEHeadDim != c.VAEWidth/c.VAEHeads {
		return fmt.Errorf("invalid Hunyuan3D VAE head dims: width=%d heads=%d head_dim=%d", c.VAEWidth, c.VAEHeads, c.VAEHeadDim)
	}
	if c.ConditionerType == "" {
		return fmt.Errorf("invalid Hunyuan3D conditioner: empty type")
	}
	if c.SchedulerSteps <= 0 {
		return fmt.Errorf("invalid Hunyuan3D scheduler steps %d", c.SchedulerSteps)
	}
	return nil
}

func (c ShapeConfig) LatentShape(batch int) ([]int, error) {
	if batch <= 0 {
		return nil, fmt.Errorf("invalid Hunyuan3D batch size %d", batch)
	}
	if err := c.Validate(); err != nil {
		return nil, err
	}
	return []int{batch, c.VAELatents, c.InChannels}, nil
}
