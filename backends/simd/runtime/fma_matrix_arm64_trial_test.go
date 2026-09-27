//go:build arm64

package simd

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// Compare the single-pass native ARM64 dispatch with exact-order scalar FMA.
// The benchmark also records the earlier clear-plus-SGEMM diagnostic path.
func BenchmarkARM64FMAMatrixPaths(b *testing.B) {
	if !HasSgemmAsm {
		b.Skip("native ARM64 SGEMM unavailable")
	}
	for _, shape := range [][3]int{{64, 1, 768}, {64, 16, 768}, {256, 80, 1024}, {32, 256, 4096}} {
		m, n, k := shape[0], shape[1], shape[2]
		a, bb, dst := make([]float32, m*k), make([]float32, k*n), make([]float32, m*n)
		rng := rand.New(rand.NewSource(int64(m*1000000 + n*1000 + k)))
		for i := range a {
			a[i] = (rng.Float32() - 0.5) * 0.125
		}
		for i := range bb {
			bb[i] = (rng.Float32() - 0.5) * 0.125
		}
		name := fmt.Sprintf("m%d_n%d_k%d", m, n, k)
		b.Run(name+"/scalar", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				fmaMatrixScalar(dst, a, bb, m, n, k)
			}
		})
		b.Run(name+"/single-pass", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if !FMAMatrixF32Checked(dst, a, bb, m, n, k) {
					b.Fatal("single-pass kernel rejected")
				}
			}
		})
	}
}

func TestARM64FMAMatrixSinglePassEdges(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("native ARM64 SGEMM unavailable")
	}
	cases := []struct {
		name    string
		a, b    []float32
		m, n, k int
	}{
		{"signed-zero", []float32{float32(math.Copysign(0, -1))}, []float32{1}, 1, 1, 1},
		{"subnormal", []float32{math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32}, []float32{1, 1}, 1, 1, 2},
		{"overflow", []float32{math.MaxFloat32}, []float32{2}, 1, 1, 1},
		{"cancellation", []float32{1e15, 1, -1e15}, []float32{1, 1, 1}, 1, 1, 3},
	}
	for _, tc := range cases {
		want, got := make([]float32, tc.m*tc.n), make([]float32, tc.m*tc.n)
		fmaMatrixScalar(want, tc.a, tc.b, tc.m, tc.n, tc.k)
		if !FMAMatrixF32Checked(got, tc.a, tc.b, tc.m, tc.n, tc.k) {
			t.Fatal(tc.name, "single-pass kernel rejected")
		}
		t.Logf("%s scalar=%g (0x%x) single-pass=%g (0x%x)", tc.name, want[0], math.Float32bits(want[0]), got[0], math.Float32bits(got[0]))
		if math.Float32bits(want[0]) != math.Float32bits(got[0]) {
			t.Errorf("%s differs", tc.name)
		}
	}
}

func TestARM64FMAMatrixSinglePassShapes(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("native ARM64 SGEMM unavailable")
	}
	for _, shape := range [][3]int{{1, 1, 1}, {3, 15, 512}, {64, 1, 768}, {64, 16, 768}, {64, 80, 768}, {256, 1, 1024}, {256, 80, 1024}, {32, 256, 4096}} {
		m, n, k := shape[0], shape[1], shape[2]
		rng := rand.New(rand.NewSource(int64(m*1000000 + n*1000 + k)))
		a, b := make([]float32, m*k), make([]float32, k*n)
		for i := range a {
			a[i] = (rng.Float32() - 0.5) * 0.125
		}
		for i := range b {
			b[i] = (rng.Float32() - 0.5) * 0.125
		}
		want, got := make([]float32, m*n), make([]float32, m*n)
		fmaMatrixScalar(want, a, b, m, n, k)
		if !FMAMatrixF32Checked(got, a, b, m, n, k) {
			t.Fatalf("single-pass kernel rejected %d,%d,%d", m, n, k)
		}
		differing := 0
		var max float64
		for i, v := range got {
			if math.Float32bits(v) != math.Float32bits(want[i]) {
				differing++
				d := math.Abs(float64(v - want[i]))
				if d > max {
					max = d
				}
			}
		}
		t.Logf("m=%d n=%d k=%d bit_differences=%d/%d max_abs=%g", m, n, k, differing, len(got), max)
		if differing != 0 {
			t.Errorf("SGEMM does not preserve exact FMA order for %d,%d,%d", m, n, k)
		}
	}
}
