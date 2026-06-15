package diffusiongemma

import "fmt"

// RunVisionTowerF32 executes a pre-materialized sequence of vision layers over
// an in-place hidden buffer. It is a bounded CPU scaffold for validating graph
// composition; production multimodal readiness still requires full checkpoint
// reference fixtures and a GGUF/vision tensor source.
func RunVisionTowerF32(hidden []float32, seqLen, hiddenSize, heads, headDim int, layers []VisionLayerF32) error {
	if len(layers) == 0 {
		return fmt.Errorf("DiffusionGemma vision tower has no layers")
	}
	for i, layer := range layers {
		if err := RunVisionLayerF32(hidden, seqLen, hiddenSize, heads, headDim, layer); err != nil {
			return fmt.Errorf("DiffusionGemma vision tower layer %d: %w", i, err)
		}
	}
	return nil
}

func LoadVisionTowerF32Prefix(weights *VisionWeights, count int) ([]VisionLayerF32, error) {
	if count <= 0 {
		return nil, fmt.Errorf("DiffusionGemma vision tower prefix count must be positive, got %d", count)
	}
	if weights == nil {
		return nil, fmt.Errorf("DiffusionGemma vision tower load missing weights")
	}
	fp := weights.ForwardPlan()
	if !fp.Ready {
		return nil, fmt.Errorf("DiffusionGemma vision forward bindings incomplete: %v", fp.Missing)
	}
	if count > len(fp.Layers) {
		return nil, fmt.Errorf("DiffusionGemma vision tower prefix count=%d exceeds layers=%d", count, len(fp.Layers))
	}
	layers := make([]VisionLayerF32, 0, count)
	for i := 0; i < count; i++ {
		layer, err := LoadVisionLayerF32(weights, i)
		if err != nil {
			return nil, err
		}
		layers = append(layers, layer)
	}
	return layers, nil
}
