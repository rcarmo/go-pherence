package ideogram4

import (
	"fmt"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func conv2DGPU(in FeatureMap, w Conv2DWeights) (FeatureMap, error) {
	if !nvidia.Available() {
		return FeatureMap{}, fmt.Errorf("nvidia runtime unavailable")
	}
	out := FeatureMap{C: w.OutC, H: in.H, W: in.W, Data: make([]float32, w.OutC*in.H*in.W)}
	if err := nvidia.IdeogramConv2D(out.Data, in.Data, w.Weight, w.Bias, w.OutC, w.InC, in.H, in.W, w.KH, w.KW); err != nil {
		return FeatureMap{}, err
	}
	return out, nil
}
