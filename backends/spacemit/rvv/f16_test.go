package rvv

import (
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func refDotF16(a, b []uint16) float32 {
	var s float32
	for i := range a {
		s += half.F16ToF32(a[i]) * half.F16ToF32(b[i])
	}
	return s
}

func TestDotF16(t *testing.T) {
	// Mix positive/negative values and force an odd tail length.
	a := []uint16{0x3c00, 0x4000, 0xbe00, 0x3800, 0xc400, 0x3555, 0x4200, 0x0000, 0x3e00}
	b := []uint16{0x4000, 0xbc00, 0x3c00, 0x4400, 0x3800, 0xb555, 0x3c00, 0x7bff, 0xbe00}
	got := DotF16(a, b)
	want := refDotF16(a, b)
	if math.Abs(float64(got-want)) > 1e-3 {
		t.Fatalf("DotF16 mismatch: got %.8f want %.8f", got, want)
	}
}

func TestDotF16LongTail(t *testing.T) {
	a := make([]uint16, 257)
	b := make([]uint16, 257)
	vals := []uint16{0x3c00, 0x4000, 0x4200, 0x4400, 0xbc00, 0xbe00, 0x3800, 0x3555}
	for i := range a {
		a[i] = vals[i%len(vals)]
		b[i] = vals[(i*3+1)%len(vals)]
	}
	got := DotF16(a, b)
	want := refDotF16(a, b)
	if math.Abs(float64(got-want)) > 1e-2 {
		t.Fatalf("DotF16 long mismatch: got %.8f want %.8f diff %.8f", got, want, got-want)
	}
}

func TestGemmF16(t *testing.T) {
	const M, N, K = 7, 5, 17 // odd sizes exercise tails in both GEMM loops and RVV vl
	vals := []uint16{0x3c00, 0x4000, 0x4200, 0x4400, 0xbc00, 0xbe00, 0x3800, 0x3555}
	A := make([]uint16, M*K)
	B := make([]uint16, N*K)
	for i := range A {
		A[i] = vals[i%len(vals)]
	}
	for i := range B {
		B[i] = vals[(i*5+3)%len(vals)]
	}
	got := make([]float32, M*N)
	GemmF16(A, B, got, M, N, K)
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			want := refDotF16(A[m*K:m*K+K], B[n*K:n*K+K])
			if math.Abs(float64(got[m*N+n]-want)) > 1e-2 {
				t.Fatalf("GemmF16[%d,%d] got %.8f want %.8f", m, n, got[m*N+n], want)
			}
		}
	}
}

func TestGemmF16Outer(t *testing.T) {
	const M, N, K = 8, 16, 33
	A := make([]uint16, M*K)
	B := make([]uint16, N*K)
	for i := range A {
		A[i] = uint16(0x3800 + (i%5)<<8)
	}
	for i := range B {
		B[i] = uint16(0x3400 + (i%7)<<8)
	}
	want := make([]float32, M*N)
	got := make([]float32, M*N)
	GemmF16(A, B, want, M, N, K)
	GemmF16Outer(A, PackBF16(B, N, K), got, M, N, K, 2)
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-3 {
			t.Fatalf("GemmF16Outer[%d] got %.8f want %.8f", i, got[i], want[i])
		}
	}
}

func TestGemmF16Outer32(t *testing.T) {
	const M, N, K = 8, 32, 33
	A := make([]uint16, M*K)
	B := make([]uint16, N*K)
	for i := range A {
		A[i] = uint16(0x3800 + (i%5)<<8)
	}
	for i := range B {
		B[i] = uint16(0x3400 + (i%7)<<8)
	}
	want := make([]float32, M*N)
	got := make([]float32, M*N)
	GemmF16(A, B, want, M, N, K)
	GemmF16Outer32(A, PackBF16N32(B, N, K), got, M, N, K, 2)
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-3 {
			t.Fatalf("GemmF16Outer32[%d] got %.8f want %.8f", i, got[i], want[i])
		}
	}
}

func TestGemmF16Outer32Batch(t *testing.T) {
	const M, N, K = 8, 32, 33
	var specs []GemmF16Outer32Spec
	var wants [][]float32
	for h := 0; h < 3; h++ {
		A := make([]uint16, M*K)
		B := make([]uint16, N*K)
		for i := range A {
			A[i] = uint16(0x3800 + ((i+h)%5)<<8)
		}
		for i := range B {
			B[i] = uint16(0x3400 + ((i*3+h)%7)<<8)
		}
		want := make([]float32, M*N)
		got := make([]float32, M*N)
		GemmF16Outer32(A, PackBF16N32(B, N, K), want, M, N, K, 1)
		specs = append(specs, GemmF16Outer32Spec{A: A, Bp: PackBF16N32(B, N, K), C: got, M: M, N: N, K: K})
		wants = append(wants, want)
	}
	GemmF16Outer32Batch(4, specs...)
	for s, sp := range specs {
		for i := range sp.C {
			if math.Abs(float64(sp.C[i]-wants[s][i])) > 1e-3 {
				t.Fatalf("spec=%d i=%d got %.8f want %.8f", s, i, sp.C[i], wants[s][i])
			}
		}
	}
}

func TestGemmF16Threaded(t *testing.T) {
	const M, N, K = 16, 9, 33
	A := make([]uint16, M*K)
	B := make([]uint16, N*K)
	for i := range A {
		A[i] = uint16(0x3800 + (i%5)<<8)
	}
	for i := range B {
		B[i] = uint16(0x3400 + (i%7)<<8)
	}
	want := make([]float32, M*N)
	got := make([]float32, M*N)
	GemmF16(A, B, want, M, N, K)
	GemmF16Threaded(A, B, got, M, N, K, 4)
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-3 {
			t.Fatalf("GemmF16Threaded[%d] got %.8f want %.8f", i, got[i], want[i])
		}
	}
}

func BenchmarkDotF16_1024(b *testing.B) {
	a := make([]uint16, 1024)
	c := make([]uint16, 1024)
	for i := range a {
		a[i] = 0x3c00 // 1.0
		c[i] = 0x3800 // 0.5
	}
	b.ResetTimer()
	var s float32
	for i := 0; i < b.N; i++ {
		s += DotF16(a, c)
	}
	_ = s
}

func BenchmarkGemmF16Encoder8T(b *testing.B) {
	const M, N, K = 1500, 1280, 1280
	A := make([]uint16, M*K)
	B := make([]uint16, N*K)
	C := make([]float32, M*N)
	for i := range A {
		A[i] = 0x3c00 // 1.0
	}
	for i := range B {
		B[i] = 0x3800 // 0.5
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GemmF16Threaded(A, B, C, M, N, K, 8)
	}
}

func BenchmarkGemmF16OuterEncoder8T(b *testing.B) {
	const M, N, K = 1500, 1280, 1280
	A := make([]uint16, M*K)
	B := make([]uint16, N*K)
	C := make([]float32, M*N)
	for i := range A {
		A[i] = 0x3c00 // 1.0
	}
	for i := range B {
		B[i] = 0x3800 // 0.5
	}
	Bp := PackBF16(B, N, K)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GemmF16Outer(A, Bp, C, M, N, K, 8)
	}
}

func BenchmarkGemmF16Outer32Encoder8T(b *testing.B) {
	const M, N, K = 1500, 1280, 1280
	A := make([]uint16, M*K)
	B := make([]uint16, N*K)
	C := make([]float32, M*N)
	for i := range A {
		A[i] = 0x3c00 // 1.0
	}
	for i := range B {
		B[i] = 0x3800 // 0.5
	}
	Bp := PackBF16N32(B, N, K)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GemmF16Outer32(A, Bp, C, M, N, K, 8)
	}
}
