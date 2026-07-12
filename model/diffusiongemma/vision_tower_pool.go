package diffusiongemma

import (
	"fmt"
	"math"
)

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

// PoolVisionGridToSoftTokensF32 mirrors Gemma4VisionPooler grouping patches by
// floor(x/k),floor(y/k), where k² = input patches / output soft tokens.
func PoolVisionGridToSoftTokensF32(patchHidden []float32, patchW, patchH, hiddenSize, softTokens int) ([]float32, error) {
	patches := patchW * patchH
	if patchW <= 0 || patchH <= 0 || hiddenSize <= 0 || len(patchHidden) != patches*hiddenSize {
		return nil, fmt.Errorf("DiffusionGemma spatial vision pool invalid patch hidden len=%d grid=%dx%d hidden=%d", len(patchHidden), patchW, patchH, hiddenSize)
	}
	if softTokens <= 0 || patches%softTokens != 0 {
		return nil, fmt.Errorf("DiffusionGemma spatial vision pool cannot map grid=%dx%d to soft_tokens=%d", patchW, patchH, softTokens)
	}
	group := patches / softTokens
	k := int(math.Sqrt(float64(group)))
	if k <= 0 || k*k != group || patchW%k != 0 || patchH%k != 0 || (patchW/k)*(patchH/k) != softTokens {
		return nil, fmt.Errorf("DiffusionGemma spatial vision pool requires square kernel: grid=%dx%d soft_tokens=%d group=%d", patchW, patchH, softTokens, group)
	}
	outW, outH := patchW/k, patchH/k
	out := make([]float32, softTokens*hiddenSize)
	invGroup := float32(1.0 / float64(group))
	for oy := 0; oy < outH; oy++ {
		for ox := 0; ox < outW; ox++ {
			dst := out[(oy*outW+ox)*hiddenSize : (oy*outW+ox+1)*hiddenSize]
			for ky := 0; ky < k; ky++ {
				for kx := 0; kx < k; kx++ {
					srcPatch := (oy*k+ky)*patchW + ox*k + kx
					src := patchHidden[srcPatch*hiddenSize : (srcPatch+1)*hiddenSize]
					for i, v := range src {
						dst[i] += v * invGroup
					}
				}
			}
		}
	}
	return out, nil
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
