package ideogram4

import (
	"fmt"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func rgbClampGPU(f FeatureMap) ([]float32, error) {
	if f.C != 3 || f.H <= 0 || f.W <= 0 || len(f.Data) < 3*f.H*f.W {
		return nil, fmt.Errorf("invalid Ideogram4 GPU RGB feature map %dx%dx%d data=%d", f.C, f.H, f.W, len(f.Data))
	}
	if !nvidia.Available() {
		return nil, fmt.Errorf("nvidia runtime unavailable")
	}
	out := make([]float32, 3*f.H*f.W)
	if err := nvidia.IdeogramRGBClamp(out, f.Data, f.H*f.W); err != nil {
		return nil, err
	}
	return out, nil
}
