package ime2

import (
	"testing"
)

func refGemm(M, N, K int, A, B []int8, C []int32) {
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum int32
			for k := 0; k < K; k++ {
				sum += int32(A[i*K+k]) * int32(B[j*K+k])
			}
			C[i*N+j] = sum
		}
	}
}

func TestGemmINT8Simple(t *testing.T) {
	M, N, K := 8, 8, 16
	A := make([]int8, M*K)
	B := make([]int8, N*K)
	C_hw := make([]int32, M*N)
	C_ref := make([]int32, M*N)

	for i := range A {
		A[i] = int8((i*7 + 3) % 127 - 63)
	}
	for i := range B {
		B[i] = int8((i*13 + 11) % 127 - 63)
	}

	GemmINT8Simple(M, N, K, A, B, C_hw)
	refGemm(M, N, K, A, B, C_ref)

	errors := 0
	for i := 0; i < M*N; i++ {
		if C_hw[i] != C_ref[i] {
			if errors < 5 {
				t.Errorf("C[%d,%d]: hw=%d ref=%d", i/N, i%N, C_hw[i], C_ref[i])
			}
			errors++
		}
	}
	if errors == 0 {
		t.Logf("GemmINT8Simple %dx%dx%d: all %d elements correct!", M, N, K, M*N)
	} else {
		t.Errorf("%d/%d errors", errors, M*N)
	}
}

func TestGemmINT8Large(t *testing.T) {
	M, N, K := 32, 32, 128
	A := make([]int8, M*K)
	B := make([]int8, N*K)
	C_hw := make([]int32, M*N)
	C_ref := make([]int32, M*N)

	for i := range A {
		A[i] = int8((i*3 + 7) % 200 - 100)
	}
	for i := range B {
		B[i] = int8((i*11 + 5) % 200 - 100)
	}

	GemmINT8Simple(M, N, K, A, B, C_hw)
	refGemm(M, N, K, A, B, C_ref)

	errors := 0
	for i := 0; i < M*N; i++ {
		if C_hw[i] != C_ref[i] {
			errors++
		}
	}
	if errors == 0 {
		t.Logf("GemmINT8Simple %dx%dx%d: all %d elements correct!", M, N, K, M*N)
	} else {
		t.Errorf("%d/%d errors", errors, M*N)
	}
}

func BenchmarkGemmINT8_32x32x128(b *testing.B) {
	M, N, K := 32, 32, 128
	A := make([]int8, M*K)
	B := make([]int8, N*K)
	C := make([]int32, M*N)
	for i := range A { A[i] = int8(i % 127) }
	for i := range B { B[i] = int8(i % 127) }

	ops := int64(M) * int64(N) * int64(K) * 2 // multiply-accumulate = 2 ops
	b.SetBytes(ops)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GemmINT8Simple(M, N, K, A, B, C)
	}
}

func BenchmarkGemmINT8_128x128x256(b *testing.B) {
	M, N, K := 128, 128, 256
	A := make([]int8, M*K)
	B := make([]int8, N*K)
	C := make([]int32, M*N)
	for i := range A { A[i] = int8(i % 127) }
	for i := range B { B[i] = int8(i % 127) }

	ops := int64(M) * int64(N) * int64(K) * 2
	b.SetBytes(ops)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GemmINT8Simple(M, N, K, A, B, C)
	}
}

func BenchmarkScalarGemm_32x32x128(b *testing.B) {
	M, N, K := 32, 32, 128
	A := make([]int8, M*K)
	B := make([]int8, N*K)
	C := make([]int32, M*N)
	for i := range A { A[i] = int8(i % 127) }
	for i := range B { B[i] = int8(i % 127) }

	ops := int64(M) * int64(N) * int64(K) * 2
	b.SetBytes(ops)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		refGemm(M, N, K, A, B, C)
	}
}


func TestGemmINT8Packed(t *testing.T) {
	M, N, K := 8, 8, 16
	A := make([]int8, M*K)
	B := make([]int8, N*K)

	for i := range A {
		A[i] = int8((i*7 + 3) % 127 - 63)
	}
	for i := range B {
		B[i] = int8((i*13 + 11) % 127 - 63)
	}

	Ap := PackTiles(A, M, K)
	Bp := PackTiles(B, N, K)
	C_hw := make([]int32, M*N)
	C_ref := make([]int32, M*N)

	GemmINT8Packed(M, N, K, Ap, Bp, C_hw)
	refGemm(M, N, K, A, B, C_ref)

	errors := 0
	for i := 0; i < M*N; i++ {
		if C_hw[i] != C_ref[i] {
			if errors < 5 {
				t.Errorf("C[%d]: hw=%d ref=%d", i, C_hw[i], C_ref[i])
			}
			errors++
		}
	}
	if errors == 0 {
		t.Logf("GemmINT8Packed %dx%dx%d: all %d elements correct!", M, N, K, M*N)
	} else {
		t.Errorf("%d/%d errors", errors, M*N)
	}
}

func TestGemmINT8PackedLarge(t *testing.T) {
	M, N, K := 32, 32, 128
	A := make([]int8, M*K)
	B := make([]int8, N*K)

	for i := range A {
		A[i] = int8((i*3 + 7) % 200 - 100)
	}
	for i := range B {
		B[i] = int8((i*11 + 5) % 200 - 100)
	}

	Ap := PackTiles(A, M, K)
	Bp := PackTiles(B, N, K)
	C_hw := make([]int32, M*N)
	C_ref := make([]int32, M*N)

	GemmINT8Packed(M, N, K, Ap, Bp, C_hw)
	refGemm(M, N, K, A, B, C_ref)

	errors := 0
	for i := 0; i < M*N; i++ {
		if C_hw[i] != C_ref[i] {
			errors++
		}
	}
	if errors == 0 {
		t.Logf("GemmINT8Packed %dx%dx%d: all %d elements correct!", M, N, K, M*N)
	} else {
		t.Errorf("%d/%d errors", errors, M*N)
	}
}

func BenchmarkGemmINT8Packed_32x32x128(b *testing.B) {
	M, N, K := 32, 32, 128
	A := make([]int8, M*K)
	B := make([]int8, N*K)
	for i := range A { A[i] = int8(i % 127) }
	for i := range B { B[i] = int8(i % 127) }
	Ap := PackTiles(A, M, K)
	Bp := PackTiles(B, N, K)
	C := make([]int32, M*N)

	ops := int64(M) * int64(N) * int64(K) * 2
	b.SetBytes(ops)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GemmINT8Packed(M, N, K, Ap, Bp, C)
	}
}

func BenchmarkGemmINT8Packed_128x128x256(b *testing.B) {
	M, N, K := 128, 128, 256
	A := make([]int8, M*K)
	B := make([]int8, N*K)
	for i := range A { A[i] = int8(i % 127) }
	for i := range B { B[i] = int8(i % 127) }
	Ap := PackTiles(A, M, K)
	Bp := PackTiles(B, N, K)
	C := make([]int32, M*N)

	ops := int64(M) * int64(N) * int64(K) * 2
	b.SetBytes(ops)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GemmINT8Packed(M, N, K, Ap, Bp, C)
	}
}


func TestGemmINT8PackedParallel(t *testing.T) {
	M, N, K := 128, 128, 256
	A := make([]int8, M*K)
	B := make([]int8, N*K)

	for i := range A {
		A[i] = int8((i*3 + 7) % 200 - 100)
	}
	for i := range B {
		B[i] = int8((i*11 + 5) % 200 - 100)
	}

	Ap := PackTiles(A, M, K)
	Bp := PackTiles(B, N, K)
	C_par := make([]int32, M*N)
	C_ref := make([]int32, M*N)

	GemmINT8PackedParallel(M, N, K, Ap, Bp, C_par, 8)
	refGemm(M, N, K, A, B, C_ref)

	errors := 0
	for i := 0; i < M*N; i++ {
		if C_par[i] != C_ref[i] {
			if errors < 3 {
				t.Errorf("C[%d]: par=%d ref=%d", i, C_par[i], C_ref[i])
			}
			errors++
		}
	}
	if errors == 0 {
		t.Logf("GemmINT8PackedParallel 128x128x256 (8 threads): all %d elements correct!", M*N)
	} else {
		t.Errorf("%d/%d errors", errors, M*N)
	}
}

func BenchmarkGemmINT8PackedParallel_128x128x256_8T(b *testing.B) {
	M, N, K := 128, 128, 256
	A := make([]int8, M*K)
	B := make([]int8, N*K)
	for i := range A { A[i] = int8(i % 127) }
	for i := range B { B[i] = int8(i % 127) }
	Ap := PackTiles(A, M, K)
	Bp := PackTiles(B, N, K)
	C := make([]int32, M*N)

	ops := int64(M) * int64(N) * int64(K) * 2
	b.SetBytes(ops)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GemmINT8PackedParallel(M, N, K, Ap, Bp, C, 8)
	}
}

func BenchmarkGemmINT8PackedParallel_256x256x512_8T(b *testing.B) {
	M, N, K := 256, 256, 512
	A := make([]int8, M*K)
	B := make([]int8, N*K)
	for i := range A { A[i] = int8(i % 127) }
	for i := range B { B[i] = int8(i % 127) }
	Ap := PackTiles(A, M, K)
	Bp := PackTiles(B, N, K)
	C := make([]int32, M*N)

	ops := int64(M) * int64(N) * int64(K) * 2
	b.SetBytes(ops)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GemmINT8PackedParallel(M, N, K, Ap, Bp, C, 8)
	}
}

func BenchmarkGemmINT8PackedParallel_1024x1024x1024_8T(b *testing.B) {
	M, N, K := 1024, 1024, 1024
	A := make([]int8, M*K)
	B := make([]int8, N*K)
	for i := range A { A[i] = int8(i % 127) }
	for i := range B { B[i] = int8(i % 127) }
	Ap := PackTiles(A, M, K)
	Bp := PackTiles(B, N, K)
	C := make([]int32, M*N)

	ops := int64(M) * int64(N) * int64(K) * 2
	b.SetBytes(ops)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GemmINT8PackedParallel(M, N, K, Ap, Bp, C, 8)
	}
}
