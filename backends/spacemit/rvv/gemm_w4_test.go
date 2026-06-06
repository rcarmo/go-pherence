package rvv

import "testing"

func TestGemmW4(t *testing.T) {
	M, N, K := 8, 64, 130
	A := make([]uint8, M*K)
	W := make([]int8, N*K)
	for i := range A {
		A[i] = uint8((i * 7) % 200)
	}
	for i := range W {
		W[i] = int8((i*13)%15 - 7)
	}
	B4 := PackBW4(W, N, K)
	C := make([]int32, M*N)
	GemmU8W4(A, B4, C, M, N, K, 1)
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var r int32
			for k := 0; k < K; k++ {
				r += int32(A[m*K+k]) * int32(W[n*K+k])
			}
			if C[m*N+n] != r {
				t.Fatalf("m%d n%d got=%d ref=%d", m, n, C[m*N+n], r)
			}
		}
	}
}

func BenchmarkGemmW8_8T(b *testing.B) {
	M, N, K := 1500, 1280, 1280
	A := make([]uint8, M*K)
	W := make([]int8, N*K)
	C := make([]int32, M*N)
	Wp := PackB(W, N, K)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gemmU8Outer(A, Wp, C, M, N, K, 8)
	}
}
func BenchmarkGemmW4_8T(b *testing.B) {
	M, N, K := 1500, 1280, 1280
	A := make([]uint8, M*K)
	W := make([]int8, N*K)
	C := make([]int32, M*N)
	B4 := PackBW4(W, N, K)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GemmU8W4(A, B4, C, M, N, K, 8)
	}
}
