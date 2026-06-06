package ideogram4

import (
	"fmt"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func vaeSpatialAttentionGPU(q, k, v FeatureMap, scale float32) (FeatureMap, error) {
	if q.C != k.C || q.C != v.C || q.H != k.H || q.H != v.H || q.W != k.W || q.W != v.W {
		return FeatureMap{}, fmt.Errorf("ideogram4 GPU VAE attention shape mismatch q=%dx%dx%d k=%dx%dx%d v=%dx%dx%d", q.C, q.H, q.W, k.C, k.H, k.W, v.C, v.H, v.W)
	}
	if !nvidia.Available() {
		return FeatureMap{}, fmt.Errorf("nvidia runtime unavailable")
	}
	C, HW := q.C, q.H*q.W
	qt := make([]float32, HW*C)
	kt := make([]float32, HW*C)
	vt := make([]float32, HW*C)
	for p := 0; p < HW; p++ {
		for c := 0; c < C; c++ {
			qt[p*C+c] = q.Data[c*HW+p]
			kt[p*C+c] = k.Data[c*HW+p]
			vt[p*C+c] = v.Data[c*HW+p]
		}
	}
	outT := make([]float32, HW*C)
	if err := nvidia.IdeogramFullAttention(outT, qt, kt, vt, HW, 1, C, scale); err != nil {
		return FeatureMap{}, err
	}
	out := FeatureMap{C: C, H: q.H, W: q.W, Data: make([]float32, C*HW)}
	for p := 0; p < HW; p++ {
		for c := 0; c < C; c++ {
			out.Data[c*HW+p] = outT[p*C+c]
		}
	}
	return out, nil
}
