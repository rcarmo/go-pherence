package ime2

import "testing"

func BenchmarkGemmINT8Packed_2048x4x1024(b *testing.B) {
	M, N, K := 2048, 4, 1024
	A := make([]int8, M*K)
	Bp := make([]int8, N*K) // pre-packed B (tiny)
	for i := range A {
		A[i] = int8(i % 64)
	}
	for i := range Bp {
		Bp[i] = int8(i % 64)
	}
	Ap := PackTiles(A, M, K)
	BpPacked := PackTiles(Bp, N, K)
	C := make([]int32, M*N)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GemmINT8Packed(M, N, K, Ap, BpPacked, C)
	}
}

func BenchmarkGemmINT8Packed_3072x4x1024(b *testing.B) {
	M, N, K := 3072, 4, 1024
	Ap := PackTiles(make([]int8, M*K), M, K)
	Bp := PackTiles(make([]int8, N*K), N, K)
	C := make([]int32, M*N)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GemmINT8Packed(M, N, K, Ap, Bp, C)
	}
}
