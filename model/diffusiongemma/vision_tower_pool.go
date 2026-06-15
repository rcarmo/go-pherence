package diffusiongemma

import "fmt"

// RunVisionTowerAndPoolF32 executes a supplied vision tower prefix and pools the
// resulting patch sequence to soft-token hidden states. It deliberately accepts
// already-materialized layers so tests can validate tower→pool graph semantics
// without loading/running the full image checkpoint path by default.
func RunVisionTowerAndPoolF32(patchHidden []float32, patches, hiddenSize, heads, headDim, softTokens int, layers []VisionLayerF32) ([]float32, error) {
	if patches <= 0 || hiddenSize <= 0 || len(patchHidden) != patches*hiddenSize {
		return nil, fmt.Errorf("DiffusionGemma vision tower pool invalid patch hidden len=%d patches=%d hidden=%d", len(patchHidden), patches, hiddenSize)
	}
	if err := RunVisionTowerF32(patchHidden, patches, hiddenSize, heads, headDim, layers); err != nil {
		return nil, err
	}
	return PoolVisionPatchesToSoftTokensF32(patchHidden, patches, hiddenSize, softTokens)
}

func PoolVisionPatchesToSoftTokensF32(patchHidden []float32, patches, hiddenSize, softTokens int) ([]float32, error) {
	if patches <= 0 || hiddenSize <= 0 || len(patchHidden) != patches*hiddenSize {
		return nil, fmt.Errorf("DiffusionGemma vision pool invalid patch hidden len=%d patches=%d hidden=%d", len(patchHidden), patches, hiddenSize)
	}
	if softTokens <= 0 || patches < softTokens || patches%softTokens != 0 {
		return nil, fmt.Errorf("DiffusionGemma vision pool cannot map patches=%d to soft_tokens=%d", patches, softTokens)
	}
	group := patches / softTokens
	out := make([]float32, softTokens*hiddenSize)
	invGroup := float32(1.0 / float64(group))
	for tok := 0; tok < softTokens; tok++ {
		dst := out[tok*hiddenSize : (tok+1)*hiddenSize]
		for j := 0; j < group; j++ {
			src := patchHidden[(tok*group+j)*hiddenSize : (tok*group+j+1)*hiddenSize]
			for i, v := range src {
				dst[i] += v * invGroup
			}
		}
	}
	return out, nil
}
