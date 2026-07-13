package nvidia

import (
	"math"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func TestDevRoPENeoXRotateHalfParity(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	const heads, headDim, pos = 2, 8, 1
	x := make([]float32, heads*headDim)
	for i := range x {
		x[i] = float32(i-7) * 0.13
	}
	// Two positions, four interleaved cosine/sine pairs per position.
	freqs := make([]float32, 2*headDim)
	for p := 0; p < 2; p++ {
		for i := 0; i < headDim/2; i++ {
			angle := float64((p+1)*(i+1)) * 0.17
			freqs[p*headDim+i*2] = float32(math.Cos(angle))
			freqs[p*headDim+i*2+1] = float32(math.Sin(angle))
		}
	}
	want := append([]float32(nil), x...)
	simd.ApplyRoPE(want, freqs, pos, heads, headDim)
	xGPU, tableGPU := NewDevBufFrom(append([]float32(nil), x...)), NewDevBufFrom(freqs)
	defer xGPU.Free()
	defer tableGPU.Free()
	if !DevRoPE(xGPU, tableGPU, pos, heads, headDim) {
		t.Fatal("DevRoPE returned fallback")
	}
	got := xGPU.Data()
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > 2e-6 {
			t.Fatalf("RoPE[%d]=%g want=%g diff=%g", i, got[i], want[i], diff)
		}
	}
}
