package ideogram4

import (
	"fmt"
	"os"
	"strings"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func gpuNormEnabled() bool {
	if gpuDisabledByK3() {
		return false
	}
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_NORM")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func gpuNormStrict() bool {
	if gpuDisabledByK3() {
		return false
	}
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_NORM_STRICT")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func layerNormNoAffineGPU(dst, x []float32, eps float32) error {
	if len(dst) != len(x) || len(x) == 0 {
		return fmt.Errorf("invalid Ideogram4 GPU LayerNorm row dst=%d x=%d", len(dst), len(x))
	}
	if gpuDisabledByK3() {
		return fmt.Errorf("layer_norm_no_affine: %w", errK3GPUDisabled)
	}
	if !nvidia.Available() {
		return fmt.Errorf("nvidia runtime unavailable: norm")
	}
	return nvidia.IdeogramLayerNormNoAffine(dst, x, 1, len(x), eps)
}

func rmsNormRowsWeightedGPU(dst, x, weight, scale []float32, rows, cols int, eps float32) error {
	if rows <= 0 || cols <= 0 || len(dst) < rows*cols || len(x) < rows*cols || len(weight) < cols {
		return fmt.Errorf("invalid Ideogram GPU RMSNorm rows dst=%d x=%d weight=%d rows=%d cols=%d", len(dst), len(x), len(weight), rows, cols)
	}
	if gpuDisabledByK3() {
		return fmt.Errorf("rms_rows: %w", errK3GPUDisabled)
	}
	if !nvidia.Available() {
		return fmt.Errorf("nvidia runtime unavailable: norm")
	}
	return nvidia.IdeogramRMSNormRows(dst, x, weight, scale, rows, cols, eps)
}

func rmsNormWeightedGPU(dst, x, weight []float32, eps float32) error {
	if len(dst) != len(x) || len(weight) != len(x) || len(x) == 0 {
		return fmt.Errorf("invalid Ideogram4 GPU RMSNorm row dst=%d x=%d weight=%d", len(dst), len(x), len(weight))
	}
	if gpuDisabledByK3() {
		return fmt.Errorf("rms_weighted: %w", errK3GPUDisabled)
	}
	if !nvidia.Available() {
		return fmt.Errorf("nvidia runtime unavailable: norm")
	}
	return nvidia.F32RMSNorm(dst, x, weight, eps)
}

func adalnTransformGPU(mod []float32, emb int) error {
	if gpuDisabledByK3() {
		return fmt.Errorf("adaln_transform: %w", errK3GPUDisabled)
	}
	if !nvidia.Available() {
		return fmt.Errorf("nvidia runtime unavailable: norm")
	}
	return nvidia.IdeogramAdaLNTransform(mod, emb)
}

func gatedResidualRowsGPU(hidden, update, gate []float32, rows, cols int) error {
	if rows <= 0 || cols <= 0 || len(hidden) < rows*cols || len(update) < rows*cols || len(gate) < cols {
		return fmt.Errorf("invalid Ideogram GPU gated residual rows hidden=%d update=%d gate=%d rows=%d cols=%d", len(hidden), len(update), len(gate), rows, cols)
	}
	if gpuDisabledByK3() {
		return fmt.Errorf("gated_residual: %w", errK3GPUDisabled)
	}
	if !nvidia.Available() {
		return fmt.Errorf("nvidia runtime unavailable: norm")
	}
	return nvidia.IdeogramGatedResidualRows(hidden, update, gate, rows, cols)
}
