package ideogram4

import (
	"fmt"
	"os"
	"strings"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func gpuVAEEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_VAE")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func gpuVAEStrict() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_VAE_STRICT")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func latentDenormGPU(latents []float32, channels int) error {
	want := len(latentScale)
	if len(latentShift) != want || channels != want {
		return fmt.Errorf("ideogram4 GPU latent denorm channels=%d want %d", channels, want)
	}
	if !nvidia.Available() {
		return fmt.Errorf("nvidia runtime unavailable")
	}
	return nvidia.IdeogramLatentDenorm(latents, latentScale[:], latentShift[:], channels)
}
