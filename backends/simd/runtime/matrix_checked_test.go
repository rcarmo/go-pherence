package simd

import (
	"fmt"
	"math"
	"reflect"
	"testing"
)

func TestMatMulAllTransposeCombos(t *testing.T) {
	for _, shape := range [][3]int{{3, 5, 7}, {5, 17, 9}, {9, 3, 11}} {
		m, n, k := shape[0], shape[1], shape[2]
		logicalA := randFloats(m*k, int64(1000+m*13+n*17+k*19))
		logicalB := randFloats(k*n, int64(2000+m*23+n*29+k*31))
		want := refMatMulLogical(logicalA, logicalB, m, n, k)
		for _, transposeA := range []bool{false, true} {
			for _, transposeB := range []bool{false, true} {
				name := fmt.Sprintf("m%d_n%d_k%d_ta%v_tb%v", m, n, k, transposeA, transposeB)
				t.Run(name, func(t *testing.T) {
					a := packMatMulA(logicalA, m, k, transposeA)
					b := packMatMulB(logicalB, k, n, transposeB)
					need := m * n
					storage := make([]float32, need+3)
					for i := range storage {
						storage[i] = 12345
					}
					dst := storage[1 : 1+need+1]
					if !MatMul(dst, a, b, m, n, k, transposeA, transposeB) {
						t.Fatal("MatMul rejected valid inputs")
					}
					got := dst[:need]
					for i := range want {
						if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-4 || math.IsNaN(diff) {
							t.Fatalf("index=%d got=%g want=%g diff=%g", i, got[i], want[i], diff)
						}
					}
					if storage[0] != 12345 || storage[len(storage)-1] != 12345 || dst[need] != 12345 {
						t.Fatal("MatMul mutated dst tail/canaries")
					}
				})
			}
		}
	}
}

func TestMatMulRejectsInvalidWithoutWrites(t *testing.T) {
	const (
		m = 2
		n = 3
		k = 4
	)
	needDst := m * n
	needA := m * k
	needB := k * n
	validA := randFloats(needA, 1)
	validB := randFloats(needB, 2)

	for _, tc := range []struct {
		name string
		mk   func() ([]float32, []float32, []float32, int, int, int, bool, bool)
	}{
		{
			name: "zero-dim",
			mk: func() ([]float32, []float32, []float32, int, int, int, bool, bool) {
				return []float32{9, 8, 7}, validA, validB, 0, n, k, false, false
			},
		},
		{
			name: "short-dst",
			mk: func() ([]float32, []float32, []float32, int, int, int, bool, bool) {
				return []float32{9, 8, 7, 6, 5}, validA, validB, m, n, k, false, false
			},
		},
		{
			name: "short-a",
			mk: func() ([]float32, []float32, []float32, int, int, int, bool, bool) {
				a := append([]float32(nil), validA[:needA-1]...)
				return append([]float32(nil), 1, 2, 3, 4, 5, 6), a, validB, m, n, k, true, false
			},
		},
		{
			name: "short-b",
			mk: func() ([]float32, []float32, []float32, int, int, int, bool, bool) {
				b := append([]float32(nil), validB[:needB-1]...)
				return append([]float32(nil), 1, 2, 3, 4, 5, 6), validA, b, m, n, k, false, true
			},
		},
		{
			name: "overflow-dst",
			mk: func() ([]float32, []float32, []float32, int, int, int, bool, bool) {
				maxInt := int(^uint(0) >> 1)
				return []float32{9, 8, 7}, validA, validB, maxInt, 2, 1, false, false
			},
		},
		{
			name: "alias-a",
			mk: func() ([]float32, []float32, []float32, int, int, int, bool, bool) {
				backing := make([]float32, needA+2)
				for i := range backing {
					backing[i] = float32(i + 1)
				}
				dst := backing[1 : 1+needDst]
				a := backing[:needA]
				b := append([]float32(nil), validB...)
				return dst, a, b, m, n, k, false, false
			},
		},
		{
			name: "alias-b",
			mk: func() ([]float32, []float32, []float32, int, int, int, bool, bool) {
				backing := make([]float32, needB+3)
				for i := range backing {
					backing[i] = float32(100 + i)
				}
				dst := backing[1 : 1+needDst]
				b := backing[2 : 2+needB]
				a := append([]float32(nil), validA...)
				return dst, a, b, m, n, k, false, true
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dst, a, b, m0, n0, k0, ta, tb := tc.mk()
			beforeDst := append([]float32(nil), dst...)
			beforeA := append([]float32(nil), a...)
			beforeB := append([]float32(nil), b...)
			if MatMul(dst, a, b, m0, n0, k0, ta, tb) {
				t.Fatal("accepted malformed inputs")
			}
			if !reflect.DeepEqual(dst, beforeDst) {
				t.Fatalf("dst mutated: got %v want %v", dst, beforeDst)
			}
			if !reflect.DeepEqual(a, beforeA) {
				t.Fatalf("a mutated: got %v want %v", a, beforeA)
			}
			if !reflect.DeepEqual(b, beforeB) {
				t.Fatalf("b mutated: got %v want %v", b, beforeB)
			}
		})
	}
}

func TestMatMulTransposeParity(t *testing.T) {
	const (
		m = 4
		n = 7
		k = 6
	)
	logicalA := randFloats(m*k, 11)
	logicalB := randFloats(k*n, 12)
	want := make([]float32, m*n)
	if !MatMul(want, packMatMulA(logicalA, m, k, false), packMatMulB(logicalB, k, n, false), m, n, k, false, false) {
		t.Fatal("baseline MatMul rejected valid inputs")
	}
	for _, transposeA := range []bool{false, true} {
		for _, transposeB := range []bool{false, true} {
			got := make([]float32, m*n)
			if !MatMul(got, packMatMulA(logicalA, m, k, transposeA), packMatMulB(logicalB, k, n, transposeB), m, n, k, transposeA, transposeB) {
				t.Fatal("MatMul rejected transpose parity input")
			}
			for i := range want {
				if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-4 || math.IsNaN(diff) {
					t.Fatalf("ta=%v tb=%v index=%d got=%g want=%g diff=%g", transposeA, transposeB, i, got[i], want[i], diff)
				}
			}
		}
	}
}

func BenchmarkMatMul(b *testing.B) {
	const (
		m = 32
		k = 768
		n = 576
	)
	logicalA := randFloats(m*k, 101)
	logicalB := randFloats(k*n, 102)
	for _, transposeA := range []bool{false, true} {
		for _, transposeB := range []bool{false, true} {
			name := fmt.Sprintf("m%d_k%d_n%d_ta%v_tb%v", m, k, n, transposeA, transposeB)
			a := packMatMulA(logicalA, m, k, transposeA)
			w := packMatMulB(logicalB, k, n, transposeB)
			dst := make([]float32, m*n)
			b.Run(name, func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					if !MatMul(dst, a, w, m, n, k, transposeA, transposeB) {
						b.Fatal("MatMul rejected benchmark shape")
					}
				}
			})
		}
	}
}

func refMatMulLogical(a, b []float32, m, n, k int) []float32 {
	out := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for p := 0; p < k; p++ {
				sum += a[i*k+p] * b[p*n+j]
			}
			out[i*n+j] = sum
		}
	}
	return out
}

func packMatMulA(logical []float32, m, k int, transposeA bool) []float32 {
	if !transposeA {
		return append([]float32(nil), logical...)
	}
	out := make([]float32, m*k)
	for i := 0; i < m; i++ {
		for p := 0; p < k; p++ {
			out[p*m+i] = logical[i*k+p]
		}
	}
	return out
}

func packMatMulB(logical []float32, k, n int, transposeB bool) []float32 {
	if !transposeB {
		return append([]float32(nil), logical...)
	}
	out := make([]float32, k*n)
	for p := 0; p < k; p++ {
		for j := 0; j < n; j++ {
			out[j*k+p] = logical[p*n+j]
		}
	}
	return out
}
