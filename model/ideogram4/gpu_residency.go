package ideogram4

import (
	"os"
	"strings"
)

type gpuResidencyMode int

const (
	gpuResidencyPersistent gpuResidencyMode = iota
	gpuResidencyPhase
	gpuResidencyStream
)

// gpuResidencyPolicy returns the coarse 12GB-VRAM policy for opt-in Ideogram GPU
// execution. persistent keeps cached linears until ReleaseGPU; phase frees model
// caches between Qwen/DiT/VAE phases; stream also disables long-lived FP8 reuse
// at orchestration boundaries and relies on per-call temporary uploads.
func gpuResidencyPolicy() gpuResidencyMode {
	switch strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_RESIDENCY"))) {
	case "phase", "phased", "12gb":
		return gpuResidencyPhase
	case "stream", "streaming", "minimal":
		return gpuResidencyStream
	default:
		return gpuResidencyPersistent
	}
}

func gpuReleaseAfterPhase() bool {
	m := gpuResidencyPolicy()
	return m == gpuResidencyPhase || m == gpuResidencyStream
}
