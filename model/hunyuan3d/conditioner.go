package hunyuan3d

import "fmt"

// ConditionerConfig is the metadata needed before implementing DINO/CLIP image
// encoders. It describes the preprocessed image tensor and the expected patch
// token count for ViT-style encoders.
type ConditionerConfig struct {
	Type      string
	ImageSize int
	PatchSize int
	Channels  int
}

func DefaultConditionerConfig(condType string) ConditionerConfig {
	// DINOv2-style encoders commonly use 14x14 patches; 518 keeps a whole
	// 37x37 patch grid. The upstream image processor may still emit 512px images
	// for other encoders/configs, so callers can override this metadata.
	return ConditionerConfig{Type: condType, ImageSize: 518, PatchSize: 14, Channels: 3}
}

func (c ConditionerConfig) Validate() error {
	if c.Type == "" {
		return fmt.Errorf("invalid Hunyuan3D conditioner: empty type")
	}
	if c.ImageSize <= 0 || c.PatchSize <= 0 || c.Channels <= 0 {
		return fmt.Errorf("invalid Hunyuan3D conditioner dims: %+v", c)
	}
	if c.ImageSize%c.PatchSize != 0 {
		return fmt.Errorf("invalid Hunyuan3D conditioner patch grid: image_size=%d patch_size=%d", c.ImageSize, c.PatchSize)
	}
	return nil
}

func (c ConditionerConfig) ImageTensorShape(batch int) ([]int, error) {
	if batch <= 0 {
		return nil, fmt.Errorf("invalid Hunyuan3D conditioner batch size %d", batch)
	}
	if err := c.Validate(); err != nil {
		return nil, err
	}
	return []int{batch, c.Channels, c.ImageSize, c.ImageSize}, nil
}

func (c ConditionerConfig) PatchGrid() (int, int, error) {
	if err := c.Validate(); err != nil {
		return 0, 0, err
	}
	g := c.ImageSize / c.PatchSize
	return g, g, nil
}

func (c ConditionerConfig) PatchTokenCount(includeClassToken bool) (int, error) {
	gw, gh, err := c.PatchGrid()
	if err != nil {
		return 0, err
	}
	n := gw * gh
	if includeClassToken {
		n++
	}
	return n, nil
}

func ConditionerFromShapeConfig(shape ShapeConfig) (ConditionerConfig, error) {
	cfg := DefaultConditionerConfig(shape.ConditionerType)
	return cfg, cfg.Validate()
}
