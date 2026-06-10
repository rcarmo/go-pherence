package ideogram4

import (
	"fmt"
	"os"
	"strings"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func gpuNormEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_NORM")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func gpuNormStrict() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_NORM_STRICT")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func layerNormNoAffineGPU(dst, x []float32, eps float32) error {
	if len(dst) != len(x) || len(x) == 0 {
		return fmt.Errorf("invalid Ideogram4 GPU LayerNorm row dst=%d x=%d", len(dst), len(x))
	}
	if !nvidia.Available() {
		return fmt.Errorf("nvidia runtime unavailable")
	}
	return nvidia.IdeogramLayerNormNoAffine(dst, x, 1, len(x), eps)
}

func rmsNormRowsWeightedGPU(dst, x, weight, scale []float32, rows, cols int, eps float32) error {
	if rows <= 0 || cols <= 0 || len(dst) < rows*cols || len(x) < rows*cols || len(weight) < cols {
		return fmt.Errorf("invalid Ideogram GPU RMSNorm rows dst=%d x=%d weight=%d rows=%d cols=%d", len(dst), len(x), len(weight), rows, cols)
	}
	if !nvidia.Available() {
		return fmt.Errorf("nvidia runtime unavailable")
	}
	return nvidia.IdeogramRMSNormRows(dst, x, weight, scale, rows, cols, eps)
}

func rmsNormWeightedGPU(dst, x, weight []float32, eps float32) error {
	if len(dst) != len(x) || len(weight) != len(x) || len(x) == 0 {
		return fmt.Errorf("invalid Ideogram4 GPU RMSNorm row dst=%d x=%d weight=%d", len(dst), len(x), len(weight))
	}
	if !nvidia.Available() {
		return fmt.Errorf("nvidia runtime unavailable")
	}
	return nvidia.F32RMSNorm(dst, x, weight, eps)
}

func adalnTransformGPU(mod []float32, emb int) error {
	if !nvidia.Available() {
		return fmt.Errorf("nvidia runtime unavailable")
	}
	return nvidia.IdeogramAdaLNTransform(mod, emb)
}

func gatedResidualRowsGPU(hidden, update, gate []float32, rows, cols int) error {
	if rows <= 0 || cols <= 0 || len(hidden) < rows*cols || len(update) < rows*cols || len(gate) < cols {
		return fmt.Errorf("invalid Ideogram GPU gated residual rows hidden=%d update=%d gate=%d rows=%d cols=%d", len(hidden), len(update), len(gate), rows, cols)
	}
	if !nvidia.Available() {
		return fmt.Errorf("nvidia runtime unavailable")
	}
	return nvidia.IdeogramGatedResidualRows(hidden, update, gate, rows, cols)
}
