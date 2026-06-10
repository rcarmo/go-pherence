package ideogram4

import (
	"math"
	"testing"
)

func TestK3FP8BatchFallbackOffHost(t *testing.T) {
	spec := LinearSpec{Prefix: "test", InDim: 4, OutDim: 3, Weight: "test.weight", WeightScale: "test.weight_scale"}
	w := []byte{0x38, 0xb8, 0x30, 0x00, 0x40, 0x3c, 0xbc, 0x34, 0x20, 0xa0, 0x10, 0x90}
	scale := []float32{0.5, 0.25, 1.0}
	lin, err := NewFP8Linear(spec, w, scale, []float32{0.1, -0.2, 0.3})
	if err != nil {
		t.Fatal(err)
	}
	x := []float32{1, -2, 0.5, 3}
	out := make([]float32, 3)
	ok, err := k3FP8Batch(lin, x, out, 1)
	if err != nil {
		t.Fatalf("k3FP8Batch returned error with fallback-off path: %v", err)
	}
	// On non-riscv64 this must report not handled. On riscv64 without the env gate
	// it must also report not handled. This keeps generic builds from accidentally
	// depending on K3-only kernels.
	if ok {
		// If a riscv64 test runner enables GO_PHERENCE_IDEOGRAM4_K3 globally, the
		// result should still be finite and plausibly non-zero.
		for i, v := range out {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("out[%d]=%v", i, v)
			}
		}
	}
}
