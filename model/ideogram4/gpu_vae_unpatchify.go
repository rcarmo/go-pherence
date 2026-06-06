package ideogram4

import (
	"fmt"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func unpatchifyLatentsGPU(tokens []float32, gridH, gridW, inChannels, latentChannels, patchH, patchW int) (FeatureMap, error) {
	if !nvidia.Available() {
		return FeatureMap{}, fmt.Errorf("nvidia runtime unavailable")
	}
	H := gridH * patchH
	W := gridW * patchW
	out := FeatureMap{C: latentChannels, H: H, W: W, Data: make([]float32, latentChannels*H*W)}
	if err := nvidia.IdeogramUnpatchify(out.Data, tokens, gridH, gridW, inChannels, latentChannels, patchH, patchW); err != nil {
		return FeatureMap{}, err
	}
	return out, nil
}
