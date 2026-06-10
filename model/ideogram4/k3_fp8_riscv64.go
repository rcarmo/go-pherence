//go:build riscv64

package ideogram4

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/rcarmo/go-pherence/backends/simd/quant/fp8"
	"github.com/rcarmo/go-pherence/backends/spacemit/rvv"
	"github.com/rcarmo/go-pherence/half"
)

func k3Enabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_K3")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func k3Threads() int {
	if s := strings.TrimSpace(os.Getenv("GO_PHERENCE_IDEOGRAM4_K3_THREADS")); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 {
			return n
		}
	}
	return 8
}

func k3FP8Batch(f *FP8Linear, x, out []float32, batch int) (bool, error) {
	if f == nil || !k3Enabled() || batch <= 0 {
		return false, nil
	}
	inDim, outDim := f.weight.InDim, f.weight.OutDim
	if len(x) < batch*inDim || len(out) < batch*outDim {
		return true, fmt.Errorf("ideogram4 K3 FP8 linear %q invalid buffers x=%d/%d out=%d/%d", f.spec.Prefix, len(x), batch*inDim, len(out), batch*outDim)
	}
	// First K3 SIMD coverage path: decode FP8 weights to fp16 rows and convert
	// F32 activations to fp16, then use the existing RVV/Zvfh fp16 GEMM kernels.
	// This is intentionally conservative and correctness-oriented; later K3 work
	// should replace this with resident packed FP8→int8/IME2 kernels.
	A := make([]uint16, batch*inDim)
	rvv.F32ToF16RVV(A, x[:batch*inDim])
	B := make([]uint16, outDim*inDim)
	for r := 0; r < outDim; r++ {
		scale := f.weight.Scale[0]
		if len(f.weight.Scale) != 1 {
			scale = f.weight.Scale[r]
		}
		wb := f.weight.Weight[r*inDim : (r+1)*inDim]
		bb := B[r*inDim : (r+1)*inDim]
		for c := 0; c < inDim; c++ {
			bb[c] = half.F32ToF16(fp8.DecodeE4M3(wb[c]) * scale)
		}
	}
	rvv.GemmF16Threaded(A, B, out[:batch*outDim], batch, outDim, inDim, k3Threads())
	if f.weight.Bias != nil {
		for b := 0; b < batch; b++ {
			row := out[b*outDim : (b+1)*outDim]
			for i, bias := range f.weight.Bias {
				row[i] += bias
			}
		}
	}
	return true, nil
}
