package rvv

import "testing"

func TestGemmOuter(t *testing.T) {
	M, N, K := 8, 64, 130
	A := make([]int8, M*K)
	B := make([]int8, N*K)
	for i := range A {
		A[i] = int8((i*7)%255 - 128)
	}
	for i := range B {
		B[i] = int8((i*13)%255 - 128)
	}
	Bp := PackB(B, N, K)
	C := make([]int32, M*N)
	GemmI8Outer(A, Bp, C, M, N, K, 1)
	want := refGemm(A, B, M, N, K)
	for i := range C {
		if C[i] != want[i] {
			t.Fatalf("idx %d got=%d want=%d", i, C[i], want[i])
		}
	}
}

func BenchmarkGemmOuter8T(b *testing.B) {
	M, N, K := 1500, 1280, 1280
	A := make([]int8, M*K)
	B2 := make([]int8, N*K)
	C := make([]int32, M*N)
	for i := range A {
		A[i] = int8(i)
	}
	for i := range B2 {
		B2[i] = int8(i * 3)
	}
	Bp := PackB(B2, N, K)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GemmI8Outer(A, Bp, C, M, N, K, 8)
	}
}
