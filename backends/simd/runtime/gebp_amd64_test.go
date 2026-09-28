//go:build amd64

package simd

import (
	"fmt"
	"math"
	"slices"
	"testing"
	"unsafe"
)

// This kernel already uses FMA. Keep its per-output reduction sequence and its
// separately rounded alpha multiply/C addition, including every K-loop tail.
func TestGebpAMD64ReductionOrder(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("AVX2/FMA unavailable")
	}
	for _, k := range []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32, 33, 127, 128, 129, 1024, 3584} {
		for _, alpha := range []float32{0, 1, -.75} {
			t.Run(fmt.Sprintf("k=%d/alpha=%g", k, alpha), func(t *testing.T) {
				lda, ldc := k+3, 19
				aStore, bStore, cStore := make([]float32, 6*lda+2), make([]float32, max(1, k*16)+2), make([]float32, 6*ldc+2)
				a, b, c := aStore[1:len(aStore)-1], bStore[1:len(bStore)-1], cStore[1:len(cStore)-1]
				seed := uint32(97)
				for _, s := range [][]float32{aStore, bStore, cStore} {
					for i := range s {
						seed = seed*1664525 + 1013904223
						// Similar-magnitude F32 operands: float64 has enough bits
						// for an independent correctly rounded F32 FMA oracle.
						s[i] = math.Float32frombits((seed & 0x807fffff) | 0x3f000000)
					}
				}
				savedA, savedB, want := slices.Clone(aStore), slices.Clone(bStore), slices.Clone(cStore)
				for i := 0; i < 6; i++ {
					for j := 0; j < 16; j++ {
						sum := float32(0)
						for p := 0; p < k; p++ {
							sum = float32(math.FMA(float64(a[i*lda+p]), float64(b[p*16+j]), float64(sum)))
						}
						product := float32(float64(alpha) * float64(sum))
						want[1+i*ldc+j] = float32(float64(want[1+i*ldc+j]) + float64(product))
					}
				}
				gebpMicroKernel(k, alpha, unsafe.Pointer(&a[0]), lda, unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), ldc)
				for i, v := range want {
					if math.Float32bits(cStore[i]) != math.Float32bits(v) {
						t.Fatalf("C[%d]=%08x want %08x", i, math.Float32bits(cStore[i]), math.Float32bits(v))
					}
				}
				if !slices.Equal(aStore, savedA) || !slices.Equal(bStore, savedB) {
					t.Fatal("input changed")
				}
			})
		}
	}
}

func TestGebpAMD64ExceptionalValues(t *testing.T) {
	if !HasSgemmAsm {
		t.Skip("AVX2/FMA unavailable")
	}
	for _, k := range []int{0, 1, 2, 3, 4, 5, 7, 8, 9} {
		for _, value := range []float32{0, math.Float32frombits(0x80000000), math.SmallestNonzeroFloat32, math.MaxFloat32, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())} {
			a, p, c := make([]float32, max(1, 6*k)), make([]float32, max(1, 16*k)), make([]float32, 6*16)
			for i := range a {
				a[i] = value
			}
			for i := range p {
				p[i] = -1
			}
			sum := float32(0)
			for range k {
				sum = float32(math.FMA(float64(value), -1, float64(sum)))
			}
			want := float32(float64(sum) + 0)
			gebpMicroKernel(k, 1, unsafe.Pointer(&a[0]), k, unsafe.Pointer(&p[0]), unsafe.Pointer(&c[0]), 16)
			for _, got := range c {
				if math.IsNaN(float64(want)) && math.IsNaN(float64(got)) {
					continue
				}
				if math.Float32bits(got) != math.Float32bits(want) {
					t.Fatalf("k=%d value=%v got=%08x want=%08x", k, value, math.Float32bits(got), math.Float32bits(want))
				}
			}
		}
	}
}

func BenchmarkGebpAMD64(b *testing.B) {
	if !HasSgemmAsm {
		b.Skip("AVX2/FMA unavailable")
	}
	for _, k := range []int{128, 1024, 3584} {
		b.Run(fmt.Sprintf("k%d", k), func(b *testing.B) {
			a, p, c := make([]float32, 6*k), make([]float32, 16*k), make([]float32, 6*16)
			for i := range a {
				a[i] = float32(i%17-8) / 32
			}
			for i := range p {
				p[i] = float32(i%23-11) / 64
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				gebpMicroKernel(k, 1, unsafe.Pointer(&a[0]), k, unsafe.Pointer(&p[0]), unsafe.Pointer(&c[0]), 16)
			}
		})
	}
}

// Representative projection shapes, with M chosen to fill six-row tiles.
func BenchmarkPackedGEMMProjection(b *testing.B) {
	for _, shape := range [][3]int{{42, 1024, 1024}, {42, 3584, 1024}, {42, 1024, 3584}, {126, 1024, 1024}, {510, 1024, 1024}} {
		m, n, k := shape[0], shape[1], shape[2]
		b.Run(fmt.Sprintf("m%d_n%d_k%d", m, n, k), func(b *testing.B) {
			a, p, c := make([]float32, m*k), make([]float32, n*k), make([]float32, m*n)
			for i := range a {
				a[i] = float32(i%17-8) / 32
			}
			for i := range p {
				p[i] = float32(i%23-11) / 64
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				SgemmNTPackedOnlyTo(c, a, p, m, n, k, 1, k, n)
			}
		})
	}
}
