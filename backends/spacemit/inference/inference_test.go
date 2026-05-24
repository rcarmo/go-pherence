package inference

import (
	"math"
	"math/rand"
	"testing"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
)

func TestQuantizeRoundtrip(t *testing.T) {
	src := make([]float32, 64)
	for i := range src {
		src[i] = rand.Float32()*2 - 1 // [-1, 1]
	}
	dst := make([]int8, 64)
	scale := QuantizeF32ToINT8(src, dst)

	// Check roundtrip error
	var maxErr float32
	for i := range src {
		reconstructed := float32(dst[i]) * scale
		err := float32(math.Abs(float64(src[i] - reconstructed)))
		if err > maxErr {
			maxErr = err
		}
	}
	t.Logf("Quantize roundtrip max error: %.6f (scale=%.6f)", maxErr, scale)
	if maxErr > 0.02 { // ~1% error for [-1,1] range
		t.Errorf("too much quantization error: %f", maxErr)
	}
}

func TestMatVecQ4K(t *testing.T) {
	M, K := 32, 64 // 32 output dims, 64 input dims

	// Create random F32 weights and quantize to INT8
	wF32 := make([]float32, M*K)
	for i := range wF32 {
		wF32[i] = rand.Float32()*2 - 1
	}
	wI8 := make([]int8, M*K)
	wScale := QuantizeF32ToINT8(wF32, wI8)

	// Pre-pack weights
	wPacked := ime2.PackTiles(wI8, M, K)

	// Input vector
	x := make([]float32, K)
	for i := range x {
		x[i] = rand.Float32()*2 - 1
	}

	// Compute with our function
	out := make([]float32, M)
	MatVecQ4K(M, K, wPacked, x, out, wScale)

	// Reference: F32 matmul
	ref := make([]float32, M)
	for i := 0; i < M; i++ {
		var sum float32
		for k := 0; k < K; k++ {
			sum += wF32[i*K+k] * x[k]
		}
		ref[i] = sum
	}

	// Compare (allow quantization error)
	var maxErr, avgErr float32
	for i := 0; i < M; i++ {
		err := float32(math.Abs(float64(out[i] - ref[i])))
		if err > maxErr {
			maxErr = err
		}
		avgErr += err
	}
	avgErr /= float32(M)
	t.Logf("MatVecQ4K %dx%d: maxErr=%.4f avgErr=%.4f", M, K, maxErr, avgErr)

	// Relative error should be small
	var maxRef float32
	for _, v := range ref {
		if a := float32(math.Abs(float64(v))); a > maxRef {
			maxRef = a
		}
	}
	relErr := maxErr / maxRef
	t.Logf("Relative error: %.2f%%", relErr*100)
	if relErr > 0.05 { // 5% relative error acceptable for INT8 quantization
		t.Errorf("relative error too high: %.2f%%", relErr*100)
	}
}

func BenchmarkMatVecQ4K_1024x1024(b *testing.B) {
	M, K := 1024, 1024
	wI8 := make([]int8, M*K)
	for i := range wI8 {
		wI8[i] = int8(i % 127)
	}
	wPacked := ime2.PackTiles(wI8, M, K)

	x := make([]float32, K)
	for i := range x {
		x[i] = 0.1
	}
	out := make([]float32, M)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVecQ4K(M, K, wPacked, x, out, 0.01)
	}
}
