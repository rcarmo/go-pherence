//go:build riscv64

package diffusiongemma

import (
	"os"
	"strings"

	simdrt "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/backends/spacemit/rvv"
)

func k3Enabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_K3")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

// k3Dot uses SIMD/RVV dot dispatch for Q·K scoring in attention.
func k3Dot(a, b []float32) float32 {
	if !k3Enabled() || len(a) == 0 || len(a) != len(b) {
		return dot(a, b)
	}
	return simdrt.Sdot(a, b)
}

// k3SoftmaxInPlace uses the shared SIMD softmax which has FastExp on riscv64.
func k3SoftmaxInPlace(x []float32) {
	if !k3Enabled() || len(x) == 0 {
		softmaxInPlace(x)
		return
	}
	simdrt.SoftmaxInPlace(x)
}

// k3SiLU uses the RVV polynomial SiLU approximation.
func k3SiLU(x float32) float32 {
	return rvv.FastSiLU(x)
}

// k3SaxpyV uses SIMD Saxpy for weighted V accumulation in attention.
func k3SaxpyV(w float32, v, out []float32) {
	if !k3Enabled() || len(v) == 0 || len(v) != len(out) {
		for i := range v {
			out[i] += w * v[i]
		}
		return
	}
	simdrt.Saxpy(w, v, out)
}
