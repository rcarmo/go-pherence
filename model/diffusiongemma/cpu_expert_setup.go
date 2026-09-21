package diffusiongemma

import "fmt"

// OpenCPUExpertIndex opens the per-expert FP8 tensors required by safetensor
// checkpoints whose text plan uses indexed rather than fused expert weights.
// The caller owns and must close the returned FP8TextWeights. Fused checkpoints
// return nil values because their experts are already bound in TextWeights.
func OpenCPUExpertIndex(modelDir string, shape Shape, weights *TextWeights) (*FP8ExpertIndex, *FP8TextWeights, error) {
	if weights == nil {
		return nil, nil, fmt.Errorf("DiffusionGemma CPU expert setup missing text weights")
	}
	if !weights.IndexedExperts {
		return nil, nil, nil
	}
	fp8Weights, err := OpenFP8TextWeights(modelDir, shape)
	if err != nil {
		return nil, nil, fmt.Errorf("DiffusionGemma CPU FP8 experts: %w", err)
	}
	experts := shape.NumExperts
	if experts <= 0 {
		experts = 128
	}
	index, err := BuildFP8ExpertIndex(fp8Weights, shape.TextLayers, experts)
	if err != nil {
		_ = fp8Weights.Close()
		return nil, nil, fmt.Errorf("DiffusionGemma CPU FP8 expert index: %w", err)
	}
	return index, fp8Weights, nil
}
