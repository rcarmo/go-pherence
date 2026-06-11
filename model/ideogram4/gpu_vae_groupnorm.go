package ideogram4

import (
	"fmt"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func groupNormGPU(in FeatureMap, groups int, gamma, beta []float32, eps float32) (FeatureMap, error) {
	if !nvidia.Available() {
		return FeatureMap{}, fmt.Errorf("nvidia runtime unavailable: vae_groupnorm")
	}
	out := FeatureMap{C: in.C, H: in.H, W: in.W, Data: make([]float32, len(in.Data))}
	if err := nvidia.IdeogramGroupNorm(out.Data, in.Data, gamma, beta, in.C, in.H, in.W, groups, eps); err != nil {
		return FeatureMap{}, err
	}
	return out, nil
}
