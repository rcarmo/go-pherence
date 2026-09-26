package simd

import (
	"fmt"
	"math"
	"runtime"
	"testing"
)

func TestSgemmNTPackedOnlyParity(t *testing.T) {
	const tol = 2e-4
	for _, dispatch := range []struct {
		name    string
		enabled bool
	}{
		{name: "scalar", enabled: false},
		{name: "dispatch", enabled: true},
	} {
		t.Run(dispatch.name, func(t *testing.T) {
			setHasSgemmAsmForPackedOnlyTest(t, dispatch.enabled)
			for _, m := range []int{1, gebpMR - 1, gebpMR, gebpMR + 2} {
				for _, n := range []int{16, 32, 64} {
					for _, k := range []int{1, 7, 31, 128} {
						for _, alpha := range []float32{0, 0.75, -1.25} {
							t.Run(fmt.Sprintf("m%d/n%d/k%d/a%g", m, n, k, alpha), func(t *testing.T) {
								lda := k + (m+n+k)%3 + 1
								ldb := k + (m+2*n+k)%4 + 1
								ldc := n + (m+k)%5 + 1
								aNeed := (m-1)*lda + k
								wNeed := (n-1)*ldb + k
								cNeed := (m-1)*ldc + n
								packedLen := n * k

								aStorage := make([]float32, aNeed+1)
								a := aStorage[1:]
								packedOnlyFillFinite(a, 0x1000+uint32(m*17+n*5+k))

								wStorage := make([]float32, wNeed+1)
								weights := wStorage[1:]
								packedOnlyFillFinite(weights, 0x2000+uint32(m*7+n*11+k*3))

								cStorage := make([]float32, cNeed+2)
								for i := range cStorage {
									cStorage[i] = 12345
								}
								got := cStorage[1 : 1+cNeed+1]
								packedOnlyFillFinite(got[:cNeed], 0x3000+uint32(m*13+n*3+k*19))

								packedStorage := make([]float32, packedLen+2)
								for i := range packedStorage {
									packedStorage[i] = 54321
								}
								packed := packedStorage[1 : 1+packedLen]
								if _, err := PackSgemmNTWeightsInto(weights, n, k, ldb, packed); err != nil {
									t.Fatalf("PackSgemmNTWeightsInto: %v", err)
								}

								aBefore := append([]float32(nil), a...)
								packedBefore := append([]float32(nil), packed...)
								want := packedOnlyRawReference(append([]float32(nil), got...), a, weights, m, n, k, alpha, lda, ldb, ldc)
								if !SgemmNTPackedOnlyTo(got, a, packed, m, n, k, alpha, lda, ldc) {
									t.Fatal("SgemmNTPackedOnlyTo rejected valid input")
								}
								packedOnlyRequireActiveClose(t, got, want, m, n, ldc, tol)
								if !packedOnlySlicesEqualBits(a, aBefore) || !packedOnlySlicesEqualBits(packed, packedBefore) {
									t.Fatal("mutated source operands")
								}
								for row := 0; row < m-1; row++ {
									if !packedOnlySlicesEqualBits(got[row*ldc+n:(row+1)*ldc], want[row*ldc+n:(row+1)*ldc]) {
										t.Fatal("mutated C stride padding")
									}
								}
								if cStorage[0] != 12345 || cStorage[len(cStorage)-1] != 12345 || got[cNeed] != 12345 {
									t.Fatal("wrote outside active C footprint")
								}
								if packedStorage[0] != 54321 || packedStorage[len(packedStorage)-1] != 54321 {
									t.Fatal("mutated packed guards")
								}
							})
						}
					}
				}
			}
		})
	}
}

func TestSgemmNTPackedOnlyScalarExceptionalClassification(t *testing.T) {
	setHasSgemmAsmForPackedOnlyTest(t, false)
	const (
		m = 2
		n = 16
		k = 4
	)
	lda, ldb, ldc := k+2, k+3, n+2
	aNeed := (m-1)*lda + k
	wNeed := (n-1)*ldb + k
	cNeed := (m-1)*ldc + n
	bits := []uint32{
		0,
		0x80000000,
		0x7f800000,
		0xff800000,
		0x7fc01234,
		0x7f801234,
		1,
		0x3f800000,
		0xbf800000,
	}
	a := make([]float32, aNeed)
	weights := make([]float32, wNeed)
	baseC := make([]float32, cNeed)
	packedOnlyFillBits(a, bits)
	packedOnlyFillBits(weights, bits[2:])
	packedOnlyFillBits(baseC, bits[1:])
	packed, err := PackSgemmNTWeights(weights, n, k, ldb)
	if err != nil {
		t.Fatalf("PackSgemmNTWeights: %v", err)
	}
	for _, alpha := range []float32{0, -0.75} {
		t.Run(fmt.Sprintf("alpha%g", alpha), func(t *testing.T) {
			want := packedOnlyRawReference(append([]float32(nil), baseC...), a, weights, m, n, k, alpha, lda, ldb, ldc)
			got := append([]float32(nil), baseC...)
			if !SgemmNTPackedOnlyTo(got, a, packed, m, n, k, alpha, lda, ldc) {
				t.Fatal("SgemmNTPackedOnlyTo rejected exceptional input")
			}
			packedOnlyRequireActiveExact(t, got, want, m, n, ldc)
		})
	}
}

func TestSgemmNTPackedOnlyBitwiseWithPrepackedCompleteTiles(t *testing.T) {
	setHasSgemmAsmForPackedOnlyTest(t, true)
	for _, mTiles := range []int{1, 2, 3} {
		m := mTiles * gebpMR
		for _, n := range []int{16, 32, 64} {
			for _, k := range []int{1, 7, 31, 128} {
				lda := k + (m+k)%3 + 1
				ldb := k + (n+k)%4 + 1
				ldc := n + (m+n)%5 + 1
				a := make([]float32, (m-1)*lda+k)
				weights := make([]float32, (n-1)*ldb+k)
				baseC := make([]float32, (m-1)*ldc+n)
				packedOnlyFillFinite(a, 0x4100+uint32(m*7+n*3+k))
				packedOnlyFillFinite(weights, 0x4200+uint32(m*5+n*11+k*13))
				packedOnlyFillFinite(baseC, 0x4300+uint32(m*17+n*19+k))
				packed, err := PackSgemmNTWeights(weights, n, k, ldb)
				if err != nil {
					t.Fatalf("PackSgemmNTWeights: %v", err)
				}
				for _, alpha := range []float32{1, -0.5} {
					t.Run(fmt.Sprintf("m%d/n%d/k%d/a%g", m, n, k, alpha), func(t *testing.T) {
						want := append([]float32(nil), baseC...)
						got := append([]float32(nil), baseC...)
						if !SgemmNTPrepackedTo(want, a, weights, packed, m, n, k, alpha, lda, ldb, ldc) {
							t.Fatal("SgemmNTPrepackedTo rejected valid input")
						}
						if !SgemmNTPackedOnlyTo(got, a, packed, m, n, k, alpha, lda, ldc) {
							t.Fatal("SgemmNTPackedOnlyTo rejected valid input")
						}
						packedOnlyRequireActiveExact(t, got, want, m, n, ldc)
					})
				}
			}
		}
	}
}

func TestSgemmNTPackedOnlyAllowsReadOnlyOperandOverlapAndPreservesSources(t *testing.T) {
	setHasSgemmAsmForPackedOnlyTest(t, false)
	const (
		m = 1
		n = 16
		k = 7
	)
	lda, ldb, ldc := k, k+3, n+2
	weights := make([]float32, (n-1)*ldb+k)
	packedOnlyFillFinite(weights, 0x5151)
	packed, err := PackSgemmNTWeights(weights, n, k, ldb)
	if err != nil {
		t.Fatalf("PackSgemmNTWeights: %v", err)
	}
	storage := make([]float32, len(packed)+1)
	storage[0] = 777
	copy(storage[1:], packed)
	a := storage[1 : 1+k]
	beforeStorage := append([]float32(nil), storage...)
	baseC := make([]float32, (m-1)*ldc+n)
	packedOnlyFillFinite(baseC, 0x5252)
	want := packedOnlyRawReference(append([]float32(nil), baseC...), a, weights, m, n, k, -0.75, lda, ldb, ldc)
	got := append([]float32(nil), baseC...)
	if !SgemmNTPackedOnlyTo(got, a, storage[1:], m, n, k, -0.75, lda, ldc) {
		t.Fatal("rejected valid read-only overlap between A and packed")
	}
	packedOnlyRequireActiveExact(t, got, want, m, n, ldc)
	if !packedOnlySlicesEqualBits(storage, beforeStorage) {
		t.Fatal("mutated read-only overlapping inputs")
	}
}

func TestSgemmNTPackedOnlyRejectsMalformedTransactional(t *testing.T) {
	const (
		m = 2
		n = 16
		k = 7
	)
	alpha := float32(0.5)
	lda, ldb, ldc := k+1, k+2, n+3
	a := make([]float32, (m-1)*lda+k)
	weights := make([]float32, (n-1)*ldb+k)
	baseC := make([]float32, (m-1)*ldc+n)
	packedOnlyFillFinite(a, 0x6100)
	packedOnlyFillFinite(weights, 0x6200)
	packedOnlyFillFinite(baseC, 0x6300)
	packed, err := PackSgemmNTWeights(weights, n, k, ldb)
	if err != nil {
		t.Fatalf("PackSgemmNTWeights: %v", err)
	}
	t.Run("invalid-dims", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			m    int
			n    int
			k    int
			lda  int
			ldc  int
			pack []float32
		}{
			{name: "zero-m", m: 0, n: n, k: k, lda: lda, ldc: ldc, pack: packed},
			{name: "zero-k", m: m, n: n, k: 0, lda: lda, ldc: ldc, pack: packed},
			{name: "n-not-divisible", m: m, n: 24, k: k, lda: lda, ldc: 27, pack: make([]float32, 24*k)},
			{name: "short-lda", m: m, n: n, k: k, lda: k - 1, ldc: ldc, pack: packed},
			{name: "short-ldc", m: m, n: n, k: k, lda: lda, ldc: n - 1, pack: packed},
		} {
			c := append([]float32(nil), baseC...)
			beforeC := append([]float32(nil), c...)
			beforeA := append([]float32(nil), a...)
			beforeP := append([]float32(nil), tc.pack...)
			if SgemmNTPackedOnlyTo(c, a, tc.pack, tc.m, tc.n, tc.k, alpha, tc.lda, tc.ldc) {
				t.Fatalf("accepted %s", tc.name)
			}
			if !packedOnlySlicesEqualBits(c, beforeC) || !packedOnlySlicesEqualBits(a, beforeA) || !packedOnlySlicesEqualBits(tc.pack, beforeP) {
				t.Fatalf("mutated rejected %s", tc.name)
			}
		}
	})
	t.Run("short-buffers", func(t *testing.T) {
		for _, tc := range []struct {
			name   string
			c      []float32
			a      []float32
			packed []float32
		}{
			{name: "short-a", c: append([]float32(nil), baseC...), a: a[:len(a)-1], packed: packed},
			{name: "short-c", c: append([]float32(nil), baseC[:len(baseC)-1]...), a: a, packed: packed},
			{name: "short-packed", c: append([]float32(nil), baseC...), a: a, packed: packed[:len(packed)-1]},
		} {
			beforeC := append([]float32(nil), tc.c...)
			beforeA := append([]float32(nil), tc.a...)
			beforeP := append([]float32(nil), tc.packed...)
			if SgemmNTPackedOnlyTo(tc.c, tc.a, tc.packed, m, n, k, alpha, lda, ldc) {
				t.Fatalf("accepted %s", tc.name)
			}
			if !packedOnlySlicesEqualBits(tc.c, beforeC) || !packedOnlySlicesEqualBits(tc.a, beforeA) || !packedOnlySlicesEqualBits(tc.packed, beforeP) {
				t.Fatalf("mutated rejected %s", tc.name)
			}
		}
	})
	t.Run("overflow", func(t *testing.T) {
		maxInt := int(^uint(0) >> 1)
		c := []float32{9}
		a1 := []float32{1}
		p1 := []float32{2}
		if SgemmNTPackedOnlyTo(c, a1, p1, maxInt/2+1, 16, 2, 1, 2, 16) {
			t.Fatal("accepted overflowing A footprint")
		}
		if math.Float32bits(c[0]) != math.Float32bits(9) {
			t.Fatal("mutated C on overflowing A footprint")
		}
		kOverflow := 17
		nOverflow := maxInt/kOverflow + 1
		if rem := nOverflow % gebpNR; rem != 0 {
			nOverflow += gebpNR - rem
		}
		if SgemmNTPackedOnlyTo(c, a1, p1, 1, nOverflow, kOverflow, 1, kOverflow, nOverflow) {
			t.Fatal("accepted overflowing packed footprint")
		}
		if math.Float32bits(c[0]) != math.Float32bits(9) {
			t.Fatal("mutated C on overflowing packed footprint")
		}
	})
	t.Run("overlap", func(t *testing.T) {
		overlapStorageA := make([]float32, 256)
		packedOnlyFillFinite(overlapStorageA, 0x6401)
		aOverlap := overlapStorageA[:len(a)]
		cOverlapA := overlapStorageA[1 : 1+len(baseC)]
		beforeOverlapA := append([]float32(nil), overlapStorageA...)
		beforePacked := append([]float32(nil), packed...)
		if SgemmNTPackedOnlyTo(cOverlapA, aOverlap, packed, m, n, k, alpha, lda, ldc) {
			t.Fatal("accepted C/A overlap")
		}
		if !packedOnlySlicesEqualBits(overlapStorageA, beforeOverlapA) || !packedOnlySlicesEqualBits(packed, beforePacked) {
			t.Fatal("mutated rejected C/A overlap")
		}

		overlapStorageP := make([]float32, len(packed)+1)
		packedOnlyFillFinite(overlapStorageP, 0x6402)
		packedOverlap := overlapStorageP[:len(packed)]
		cOverlapP := overlapStorageP[1 : 1+len(baseC)]
		beforeOverlapP := append([]float32(nil), overlapStorageP...)
		beforeA := append([]float32(nil), a...)
		if SgemmNTPackedOnlyTo(cOverlapP, a, packedOverlap, m, n, k, alpha, lda, ldc) {
			t.Fatal("accepted C/packed overlap")
		}
		if !packedOnlySlicesEqualBits(overlapStorageP, beforeOverlapP) || !packedOnlySlicesEqualBits(a, beforeA) {
			t.Fatal("mutated rejected C/packed overlap")
		}
	})
}

func TestSgemmNTPackedOnlyNoAllocs(t *testing.T) {
	const (
		m = gebpMR + 1
		n = 64
		k = 128
	)
	lda, ldb, ldc := k+3, k+5, n+7
	a := make([]float32, (m-1)*lda+k)
	weights := make([]float32, (n-1)*ldb+k)
	baseC := make([]float32, (m-1)*ldc+n)
	packedOnlyFillFinite(a, 0x7100)
	packedOnlyFillFinite(weights, 0x7200)
	packedOnlyFillFinite(baseC, 0x7300)
	packed, err := PackSgemmNTWeights(weights, n, k, ldb)
	if err != nil {
		t.Fatalf("PackSgemmNTWeights: %v", err)
	}
	for _, dispatch := range []struct {
		name    string
		enabled bool
	}{
		{name: "scalar", enabled: false},
		{name: "dispatch", enabled: true},
	} {
		t.Run(dispatch.name, func(t *testing.T) {
			if dispatch.enabled && runtime.GOARCH == "riscv64" {
				t.Skip("current riscv64 gebpMicroKernel allocates")
			}
			setHasSgemmAsmForPackedOnlyTest(t, dispatch.enabled)
			c := append([]float32(nil), baseC...)
			if !SgemmNTPackedOnlyTo(c, a, packed, m, n, k, 1, lda, ldc) {
				t.Fatal("warm-up rejected valid input")
			}
			if allocs := testing.AllocsPerRun(10, func() {
				copy(c, baseC)
				if !SgemmNTPackedOnlyTo(c, a, packed, m, n, k, 1, lda, ldc) {
					panic("SgemmNTPackedOnlyTo returned false")
				}
			}); allocs != 0 {
				t.Fatalf("allocs %g", allocs)
			}
		})
	}
}

func BenchmarkSgemmNTPackedOnlyTo(b *testing.B) {
	for _, shape := range []struct {
		m int
		n int
		k int
	}{
		{m: 126, n: 1024, k: 128},
		{m: 128, n: 2048, k: 256},
	} {
		b.Run(fmt.Sprintf("m%d/n%d/k%d", shape.m, shape.n, shape.k), func(b *testing.B) {
			lda, ldb, ldc := shape.k+3, shape.k+5, shape.n+7
			a := make([]float32, (shape.m-1)*lda+shape.k)
			weights := make([]float32, (shape.n-1)*ldb+shape.k)
			c := make([]float32, (shape.m-1)*ldc+shape.n)
			packedOnlyFillFinite(a, 0x8100+uint32(shape.m*3+shape.k))
			packedOnlyFillFinite(weights, 0x8200+uint32(shape.n*5+shape.k*7))
			packed, err := PackSgemmNTWeights(weights, shape.n, shape.k, ldb)
			if err != nil {
				b.Fatal(err)
			}
			b.Run("raw-prepacked", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					clear(c)
					if !SgemmNTPrepackedTo(c, a, weights, packed, shape.m, shape.n, shape.k, 1, lda, ldb, ldc) {
						b.Fatal("SgemmNTPrepackedTo rejected projection")
					}
				}
			})
			b.Run("packed-only", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					clear(c)
					if !SgemmNTPackedOnlyTo(c, a, packed, shape.m, shape.n, shape.k, 1, lda, ldc) {
						b.Fatal("SgemmNTPackedOnlyTo rejected projection")
					}
				}
			})
		})
	}
}

func setHasSgemmAsmForPackedOnlyTest(t *testing.T, enabled bool) {
	t.Helper()
	old := HasSgemmAsm
	// Never manufacture ISA availability on an unsupported host.
	HasSgemmAsm = enabled && old
	t.Cleanup(func() { HasSgemmAsm = old })
}

func packedOnlyFillFinite(dst []float32, seed uint32) {
	x := seed
	for i := range dst {
		x = x*1664525 + 1013904223
		sw := int32((x >> 9) & 0x7ff)
		dst[i] = float32(sw-1024) * 0.03125
		if i%13 == 0 {
			dst[i] = float32(math.Copysign(0, -1))
		}
	}
}

func packedOnlyFillBits(dst []float32, bits []uint32) {
	for i := range dst {
		dst[i] = math.Float32frombits(bits[i%len(bits)])
	}
}

func packedOnlyRawReference(c, a, weights []float32, m, n, k int, alpha float32, lda, ldb, ldc int) []float32 {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0)
			for p := 0; p < k; p++ {
				sum += a[i*lda+p] * weights[j*ldb+p]
			}
			c[i*ldc+j] += alpha * sum
		}
	}
	return c
}

func packedOnlyRequireActiveClose(t *testing.T, got, want []float32, m, n, ldc int, tol float64) {
	t.Helper()
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			idx := i*ldc + j
			g, w := got[idx], want[idx]
			switch {
			case math.IsNaN(float64(w)):
				if !math.IsNaN(float64(g)) {
					t.Fatalf("(%d,%d) got=%v want NaN", i, j, g)
				}
			case math.IsInf(float64(w), 0):
				if math.Float32bits(g) != math.Float32bits(w) {
					t.Fatalf("(%d,%d) got=%08x want %08x", i, j, math.Float32bits(g), math.Float32bits(w))
				}
			default:
				if math.Float32bits(g) == math.Float32bits(w) {
					continue
				}
				diff := math.Abs(float64(g - w))
				if diff > tol || math.IsNaN(diff) {
					t.Fatalf("(%d,%d) got=%g want=%g diff=%g", i, j, g, w, diff)
				}
			}
		}
	}
}

func packedOnlyRequireActiveExact(t *testing.T, got, want []float32, m, n, ldc int) {
	t.Helper()
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			idx := i*ldc + j
			if math.Float32bits(got[idx]) != math.Float32bits(want[idx]) {
				t.Fatalf("(%d,%d) got=%08x want %08x", i, j, math.Float32bits(got[idx]), math.Float32bits(want[idx]))
			}
		}
	}
}

func packedOnlySlicesEqualBits(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}
