package ideogram4

import (
	"fmt"
	"os"
	"strings"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func gpuMLPEnabled() bool {
	if gpuDisabledByK3() {
		return false
	}
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_MLP")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func gpuMLPStrict() bool {
	if gpuDisabledByK3() {
		return false
	}
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_MLP_STRICT")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func siluGPU(out, x []float32) error {
	if len(out) != len(x) || len(x) == 0 {
		return fmt.Errorf("invalid Ideogram4 GPU SiLU dst=%d x=%d", len(out), len(x))
	}
	if !nvidia.Available() {
		return fmt.Errorf("nvidia runtime unavailable: mlp")
	}
	return nvidia.F32SiLU(out, x)
}

func mulGPU(out, a, b []float32) error {
	if len(out) != len(a) || len(out) != len(b) || len(out) == 0 {
		return fmt.Errorf("invalid Ideogram4 GPU Mul out=%d a=%d b=%d", len(out), len(a), len(b))
	}
	if !nvidia.Available() {
		return fmt.Errorf("nvidia runtime unavailable: mlp")
	}
	return nvidia.F32Mul(out, a, b)
}

func siluMulGPU(out, a, b []float32) error {
	if len(out) != len(a) || len(out) != len(b) || len(out) == 0 {
		return fmt.Errorf("invalid Ideogram4 GPU SiLU*Mul out=%d a=%d b=%d", len(out), len(a), len(b))
	}
	if !nvidia.Available() {
		return fmt.Errorf("nvidia runtime unavailable: mlp")
	}
	return nvidia.F32SiLUMul(out, a, b)
}
