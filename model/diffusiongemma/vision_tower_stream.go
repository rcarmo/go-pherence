package diffusiongemma

import "fmt"

// RunVisionTowerF32StreamingPrefix materializes and executes a bounded prefix of
// real vision layers one at a time. This avoids retaining multiple large F32
// vision layers at once, which is important before attempting broader/full tower
// validation on the local checkpoint.
func RunVisionTowerF32StreamingPrefix(hidden []float32, seqLen int, shape Shape, weights *VisionWeights, count int, patchGrid ...int) error {
	if weights == nil {
		return fmt.Errorf("DiffusionGemma streaming vision tower missing weights")
	}
	if count <= 0 {
		return fmt.Errorf("DiffusionGemma streaming vision tower count must be positive, got %d", count)
	}
	if shape.VisionHiddenSize <= 0 || shape.VisionHeads <= 0 || shape.VisionHiddenSize%shape.VisionHeads != 0 {
		return fmt.Errorf("DiffusionGemma streaming vision tower invalid shape hidden=%d heads=%d", shape.VisionHiddenSize, shape.VisionHeads)
	}
	fp := weights.ForwardPlan()
	if !fp.Ready {
		return fmt.Errorf("DiffusionGemma vision forward bindings incomplete: %v", fp.Missing)
	}
	if count > len(fp.Layers) {
		return fmt.Errorf("DiffusionGemma streaming vision tower count=%d exceeds layers=%d", count, len(fp.Layers))
	}
	headDim := shape.VisionHiddenSize / shape.VisionHeads
	if len(hidden) != seqLen*shape.VisionHiddenSize {
		return fmt.Errorf("DiffusionGemma streaming vision tower hidden len=%d want %d", len(hidden), seqLen*shape.VisionHiddenSize)
	}
	for layerIndex := 0; layerIndex < count; layerIndex++ {
		layer, err := LoadVisionLayerF32(weights, layerIndex)
		if err != nil {
			return err
		}
		if len(patchGrid) != 0 {
			if len(patchGrid) != 2 || patchGrid[0] <= 0 || patchGrid[1] <= 0 || patchGrid[0]*patchGrid[1] != seqLen {
				return fmt.Errorf("DiffusionGemma streaming vision tower invalid patch grid %v for seq=%d", patchGrid, seqLen)
			}
			layer.PatchWidth, layer.PatchHeight = patchGrid[0], patchGrid[1]
			layer.RoPETheta = 10000
		}
		if err := RunVisionLayerF32(hidden, seqLen, shape.VisionHiddenSize, shape.VisionHeads, headDim, layer); err != nil {
			return fmt.Errorf("DiffusionGemma streaming vision tower layer %d: %w", layerIndex, err)
		}
	}
	return nil
}
