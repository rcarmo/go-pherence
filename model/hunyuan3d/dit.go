package hunyuan3d

import "fmt"

// DiTConfig is the denoiser metadata needed to validate tensor shapes before
// implementing Hunyuan3DDiT runtime blocks.
type DiTConfig struct {
	InChannels        int
	ContextInDim      int
	HiddenSize        int
	NumHeads          int
	HeadDim           int
	Depth             int
	DepthSingleBlocks int
	GuidanceEmbed     bool
}

func DiTFromShapeConfig(shape ShapeConfig) (DiTConfig, error) {
	cfg := DiTConfig{
		InChannels:        shape.InChannels,
		ContextInDim:      shape.ContextInDim,
		HiddenSize:        shape.HiddenSize,
		NumHeads:          shape.NumHeads,
		HeadDim:           shape.HeadDim,
		Depth:             shape.Depth,
		DepthSingleBlocks: shape.DepthSingleBlocks,
		GuidanceEmbed:     shape.GuidanceEmbed,
	}
	return cfg, cfg.Validate()
}

func (c DiTConfig) Validate() error {
	if c.InChannels <= 0 || c.ContextInDim <= 0 || c.HiddenSize <= 0 || c.NumHeads <= 0 || c.Depth < 0 || c.DepthSingleBlocks < 0 {
		return fmt.Errorf("invalid Hunyuan3D DiT dims: %+v", c)
	}
	if c.HiddenSize%c.NumHeads != 0 || c.HeadDim != c.HiddenSize/c.NumHeads {
		return fmt.Errorf("invalid Hunyuan3D DiT head dims: hidden=%d heads=%d head_dim=%d", c.HiddenSize, c.NumHeads, c.HeadDim)
	}
	return nil
}

func (c DiTConfig) LatentTokenShape(batch, tokens int) ([]int, error) {
	if batch <= 0 || tokens <= 0 {
		return nil, fmt.Errorf("invalid Hunyuan3D DiT latent shape batch=%d tokens=%d", batch, tokens)
	}
	if err := c.Validate(); err != nil {
		return nil, err
	}
	return []int{batch, tokens, c.InChannels}, nil
}

func (c DiTConfig) ContextTokenShape(batch, tokens int) ([]int, error) {
	if batch <= 0 || tokens <= 0 {
		return nil, fmt.Errorf("invalid Hunyuan3D DiT context shape batch=%d tokens=%d", batch, tokens)
	}
	if err := c.Validate(); err != nil {
		return nil, err
	}
	return []int{batch, tokens, c.ContextInDim}, nil
}

func (c DiTConfig) HiddenTokenShape(batch, tokens int) ([]int, error) {
	if batch <= 0 || tokens <= 0 {
		return nil, fmt.Errorf("invalid Hunyuan3D DiT hidden shape batch=%d tokens=%d", batch, tokens)
	}
	if err := c.Validate(); err != nil {
		return nil, err
	}
	return []int{batch, tokens, c.HiddenSize}, nil
}

func (c DiTConfig) QKVShape(batch, tokens int) ([]int, error) {
	if batch <= 0 || tokens <= 0 {
		return nil, fmt.Errorf("invalid Hunyuan3D DiT qkv shape batch=%d tokens=%d", batch, tokens)
	}
	if err := c.Validate(); err != nil {
		return nil, err
	}
	return []int{batch, tokens, 3, c.NumHeads, c.HeadDim}, nil
}

func CFGExpandedBatch(batch int, enabled bool) (int, error) {
	if batch <= 0 {
		return 0, fmt.Errorf("invalid Hunyuan3D CFG batch size %d", batch)
	}
	if enabled {
		return batch * 2, nil
	}
	return batch, nil
}
