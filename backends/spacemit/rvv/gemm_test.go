package rvv

import "testing"

func refDot(a, b []int8) int32 {
	var s int32
	for i := range a {
		s += int32(a[i]) * int32(b[i])
	}
	return s
}

func TestDotI8(t *testing.T) {
	for _, n := range []int{1, 7, 32, 33, 100, 1280} {
		a := make([]int8, n)
		b := make([]int8, n)
		for i := 0; i < n; i++ {
			a[i] = int8((i*7)%255 - 128)
			b[i] = int8((i*13)%255 - 128)
		}
		got := dotI8(&a[0], &b[0], int64(n))
		want := refDot(a, b)
		if got != want {
			t.Fatalf("n=%d got=%d want=%d", n, got, want)
		}
	}
}

func refGemm(A, B []int8, M, N, K int) []int32 {
	C := make([]int32, M*N)
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var s int32
			for k := 0; k < K; k++ {
				s += int32(A[m*K+k]) * int32(B[n*K+k])
			}
			C[m*N+n] = s
		}
	}
	return C
}

func TestGemmI8(t *testing.T) {
	M, N, K := 17, 19, 130
	A := make([]int8, M*K)
	B := make([]int8, N*K)
	for i := range A {
		A[i] = int8((i*7)%255 - 128)
	}
	for i := range B {
		B[i] = int8((i*13)%255 - 128)
	}
	C := make([]int32, M*N)
	GemmI8(A, B, C, M, N, K)
	want := refGemm(A, B, M, N, K)
	for i := range C {
		if C[i] != want[i] {
			t.Fatalf("idx %d got=%d want=%d", i, C[i], want[i])
		}
	}
}

func BenchmarkGemmEncoder(b *testing.B) {
	M, N, K := 1500, 1280, 1280 // one encoder q_proj-class matmul
	A := make([]int8, M*K)
	B2 := make([]int8, N*K)
	C := make([]int32, M*N)
	for i := range A {
		A[i] = int8(i)
	}
	for i := range B2 {
		B2[i] = int8(i * 3)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GemmI8(A, B2, C, M, N, K)
	}
}

func BenchmarkGemmEncoder8T(b *testing.B) {
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
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GemmI8Threaded(A, B2, C, M, N, K, 8)
	}
}
