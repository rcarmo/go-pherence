package ideogram4

import (
	"fmt"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func upsampleNearestGPU(in FeatureMap, factor int) (FeatureMap, error) {
	if err := in.validate(); err != nil {
		return FeatureMap{}, err
	}
	if factor <= 0 {
		return FeatureMap{}, fmt.Errorf("ideogram4 GPU upsample factor=%d", factor)
	}
	if !nvidia.Available() {
		return FeatureMap{}, fmt.Errorf("nvidia runtime unavailable")
	}
	out := FeatureMap{C: in.C, H: in.H * factor, W: in.W * factor, Data: make([]float32, in.C*in.H*factor*in.W*factor)}
	if err := nvidia.IdeogramUpsampleNearest(out.Data, in.Data, in.C, in.H, in.W, factor); err != nil {
		return FeatureMap{}, err
	}
	return out, nil
}
