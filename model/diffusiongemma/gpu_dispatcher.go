package diffusiongemma

import (
	"fmt"
	"os"

	gpu "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// GPUDispatcher offloads GEMV-heavy operations to the GPU while keeping
// control flow and sampling on the CPU. Falls back to CPUDispatcher if
// GPU SGEMM is not available.
type GPUDispatcher struct {
	ResidentLayerPrefix int
	MaxLayers           int
	TailAfterMaxLayers  bool
	LMHeadTopK          int
	Progress            bool
	SkipEviction        bool
}

func (d GPUDispatcher) RunTextForward(ctx ForwardContext, weights *TextWeights, ops ForwardOpPlan, buffers ForwardBufferPlan) (ForwardOutput, error) {
	if !gpu.SgemmReady() {
		if d.Progress {
			fmt.Fprintf(os.Stderr, "DiffusionGemma GPU dispatcher: SGEMM not ready, falling back to CPU\n")
		}
		return d.cpuFallback().RunTextForward(ctx, weights, ops, buffers)
	}

	if d.Progress {
		fmt.Fprintf(os.Stderr, "DiffusionGemma GPU dispatcher: SGEMM ready on %s\n", gpu.DeviceName())
	}

	// TODO: implement GPU-accelerated layer dispatch using gpu.Sgemm
	// and gpu.Buffer for weight upload/GEMV/download.
	// For now, fall back to CPU dispatcher.
	return d.cpuFallback().RunTextForward(ctx, weights, ops, buffers)
}

func (d GPUDispatcher) cpuFallback() CPUDispatcher {
	return CPUDispatcher{
		ResidentLayerPrefix: d.ResidentLayerPrefix,
		MaxLayers:           d.MaxLayers,
		TailAfterMaxLayers:  d.TailAfterMaxLayers,
		LMHeadTopK:          d.LMHeadTopK,
		Progress:            d.Progress,
		SkipEviction:        d.SkipEviction,
	}
}
