package simd

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

func TestNNPackedOverwriteExact(t *testing.T) {
	rng := rand.New(rand.NewPCG(19, 34))
	for _, m := range []int{1, 5, 6, 7, 12, 32} {
		for _, n := range []int{1, 15, 16, 17, 32, 63, 64, 65} {
			for _, k := range []int{1, 3, 32, 127} {
				lda, ldb, ldc := k+3, n+3, n+5
				a, b := make([]float32, m*lda), make([]float32, k*ldb)
				for i := range a {
					a[i] = rng.Float32()*2 - 1
				}
				for i := range b {
					b[i] = rng.Float32()*2 - 1
				}
				want, got := make([]float32, m*ldc), make([]float32, m*ldc)
				for i := range got {
					got[i] = 123
				}
				for row := 0; row < m; row++ {
					for col := n; col < ldc; col++ {
						want[row*ldc+col] = 123
					}
				}
				scratch := make([]float32, k*16)
				if !SgemmNNTo(want, a, b, m, n, k, 1, lda, ldb, ldc) || !SgemmNNPackedOverwriteTo(got, a, b, scratch, m, n, k, lda, ldb, ldc) {
					t.Fatal("shape")
				}
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("m%d n%d k%d index%d got%g want%g", m, n, k, i, got[i], want[i])
					}
				}
				if allocs := testing.AllocsPerRun(2, func() { SgemmNNPackedOverwriteTo(got, a, b, scratch, m, n, k, lda, ldb, ldc) }); allocs != 0 {
					t.Fatal(allocs)
				}
			}
		}
	}
}
func TestNNPackedInvalidNoMutation(t *testing.T) {
	c := []float32{123}
	if SgemmNNPackedOverwriteTo(c, []float32{1}, []float32{2}, nil, 1, 1, 1, 1, 1, 1) || c[0] != 123 {
		t.Fatal("invalid scratch mutated output")
	}
	if SgemmNNPackedOverwriteTo(c, nil, []float32{2}, make([]float32, 16), 1, 1, 1, 1, 1, 1) || c[0] != 123 {
		t.Fatal("invalid input mutated output")
	}
}
func BenchmarkNNCodecPacked(b *testing.B) {
	for _, shape := range [][3]int{{1024, 64, 896}, {128, 64, 896}, {32, 64, 224}, {32, 1024, 1024}, {32, 256, 128}} {
		for _, packed := range []bool{false, true} {
			m, n, k := shape[0], shape[1], shape[2]
			b.Run(fmt.Sprintf("m%d/n%d/k%d/packed%v", m, n, k, packed), func(b *testing.B) {
				a, bb, c, scratch := make([]float32, m*k), make([]float32, k*n), make([]float32, m*n), make([]float32, k*16)
				for i := range a {
					a[i] = float32(i%19) * .01
				}
				for i := range bb {
					bb[i] = float32(i%31) * .01
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if packed {
						SgemmNNPackedOverwriteTo(c, a, bb, scratch, m, n, k, k, n, n)
					} else {
						clear(c)
						SgemmNNTo(c, a, bb, m, n, k, 1, k, n, n)
					}
				}
			})
		}
	}
}

func TestNNPackedExceptionalValues(t *testing.T) {
	const m, n, k = 7, 17, 9 // full tile plus both tails
	values := []float32{0, float32(math.Copysign(0, -1)), 1, -1, math.SmallestNonzeroFloat32, math.MaxFloat32, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())}
	for _, v := range values {
		a, b, c, want := make([]float32, m*k), make([]float32, k*n), make([]float32, m*n), make([]float32, m*n)
		for i := range a {
			a[i] = v
		}
		for i := range b {
			b[i] = float32(i%3) - 1
		}
		SgemmNNTo(want, a, b, m, n, k, 1, k, n, n)
		if !SgemmNNPackedOverwriteTo(c, a, b, make([]float32, k*16), m, n, k, k, n, n) {
			t.Fatal("shape")
		}
		for i := range c {
			if math.IsNaN(float64(want[i])) {
				if !math.IsNaN(float64(c[i])) {
					t.Fatal("NaN classification")
				}
			} else if math.Float32bits(c[i]) != math.Float32bits(want[i]) {
				t.Fatalf("value %g index%d: %g != %g", v, i, c[i], want[i])
			}
		}
	}
}

func TestNNPackedInvalidShapes(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, shape := range [][6]int{{0, 1, 1, 1, 1, 1}, {1, -1, 1, 1, 1, 1}, {1, 1, 0, 1, 1, 1}, {1, 1, 2, 1, 1, 1}, {1, 2, 1, 1, 1, 2}, {1, 2, 1, 1, 2, 1}, {maxInt, 1, 1, maxInt, 1, 1}, {1, maxInt, 2, 2, maxInt, maxInt}, {maxInt, 1, 1, 1, 1, maxInt}} {
		c := []float32{123}
		scratch := make([]float32, 32)
		for i := range scratch {
			scratch[i] = 456
		}
		if SgemmNNPackedOverwriteTo(c, []float32{1}, []float32{1}, scratch, shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]) {
			t.Fatalf("accepted %v", shape)
		}
		if c[0] != 123 {
			t.Fatal("output mutated")
		}
		for _, v := range scratch {
			if v != 456 {
				t.Fatal("scratch mutated")
			}
		}
	}
}
