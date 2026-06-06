package rvv

import (
	"math"
	"math/rand"
	"testing"
)

func TestMatMulIntegerDequant(t *testing.T) {
	M, N, K := 8, 64, 256
	rng := rand.New(rand.NewSource(1))
	A := make([]float32, M*K)
	W := make([]float32, N*K)
	for i := range A {
		A[i] = rng.Float32()*2 - 1
	} // centered activations
	for i := range W {
		W[i] = rng.Float32()*0.4 - 0.2
	} // small weights
	Wp, wScale, wColSum := QuantizeWeightsSym(W, N, K)
	C := make([]float32, M*N)
	MatMulIntegerDequant(A, Wp, wScale, wColSum, C, M, N, K, 1)
	// f32 reference
	var maxRel, denom float64
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var s float32
			for k := 0; k < K; k++ {
				s += A[m*K+k] * W[n*K+k]
			}
			diff := math.Abs(float64(C[m*N+n] - s))
			if math.Abs(float64(s)) > denom {
				denom = math.Abs(float64(s))
			}
			if diff > maxRel {
				maxRel = diff
			}
		}
	}
	rel := maxRel / denom
	t.Logf("max abs err=%.4f, rel=%.4f", maxRel, rel)
	if rel > 0.05 {
		t.Fatalf("int8 GEMM dequant too far from f32: rel=%.4f", rel)
	}
}

func BenchmarkMatMulIntegerDequant8T(b *testing.B) {
	M, N, K := 1500, 1280, 1280
	A := make([]float32, M*K)
	W := make([]float32, N*K)
	for i := range A {
		A[i] = float32(i%17) - 8
	}
	for i := range W {
		W[i] = float32(i%9)*0.05 - 0.2
	}
	Wp, wScale, wColSum := QuantizeWeightsSym(W, N, K)
	C := make([]float32, M*N)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulIntegerDequant(A, Wp, wScale, wColSum, C, M, N, K, 8)
	}
}
