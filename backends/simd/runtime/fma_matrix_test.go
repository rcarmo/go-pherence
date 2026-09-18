package simd

import (
	"math"
	"reflect"
	"testing"
)

func TestFMAMatrixExactOrderAndBounds(t *testing.T) {
	for _, m := range []int{1, 2, 3, 8} {
		for _, n := range []int{1, 7, 8, 9, 31, 32, 33, 63, 64, 65} {
			for _, k := range []int{1, 3, 9, 32, 251} {
				a, b := make([]float32, m*k), make([]float32, k*n)
				for i := range a {
					a[i] = float32(math.Sin(float64(i) * .031))
				}
				for i := range b {
					b[i] = float32(math.Cos(float64(i) * .053))
				}
				want := make([]float32, m*n)
				fmaMatrixScalar(want, a, b, m, n, k)
				for off := 0; off < 3; off++ {
					storage := make([]float32, m*n+off+2)
					for i := range storage {
						storage[i] = 12345
					}
					dst := storage[off+1 : off+1+m*n]
					if !FMAMatrixF32Checked(dst, a, b, m, n, k) {
						t.Fatal("reject", m, n, k)
					}
					for i, v := range dst {
						if math.Float32bits(v) != math.Float32bits(want[i]) {
							t.Fatal("order", m, n, k, i, v, want[i])
						}
					}
					for i, v := range storage {
						if (i < off+1 || i >= off+1+m*n) && v != 12345 {
							t.Fatal("canary")
						}
					}
				}
			}
		}
	}
}
func TestFMAMatrixRejectsBeforeWrites(t *testing.T) {
	for _, kind := range []string{"short", "long", "aliasA", "aliasB", "nan", "infinite", "zero", "bound", "overflow"} {
		dst, a, b := []float32{7, 8, 9, 10}, []float32{1, 2, 3, 4}, []float32{5, 6, 7, 8}
		m, n, k := 2, 2, 2
		switch kind {
		case "short":
			a = a[:3]
		case "long":
			b = append(b, 1)
		case "aliasA":
			a = dst
		case "aliasB":
			b = dst
		case "nan":
			a[0] = float32(math.NaN())
		case "infinite":
			b[0] = float32(math.Inf(1))
		case "zero":
			k = 0
		case "bound":
			m = 257
		case "overflow":
			n = int(^uint(0) >> 1)
		}
		before := append([]float32(nil), dst...)
		if FMAMatrixF32Checked(dst, a, b, m, n, k) || !reflect.DeepEqual(dst, before) {
			t.Fatal(kind)
		}
	}
	dst, a, b := make([]float32, 64), make([]float32, 64), make([]float32, 64)
	if n := testing.AllocsPerRun(10, func() {
		if !FMAMatrixF32Checked(dst, a, b, 8, 8, 8) {
			panic("reject")
		}
	}); n != 0 {
		t.Fatal(n)
	}
	// Admitted bounds and model reduction widths, without a giant cubic test.
	for _, shape := range [][3]int{{256, 1, 9}, {1, 256, 9}, {1, 1, 4096}, {3, 33, 2304}} {
		m, n, k := shape[0], shape[1], shape[2]
		a, b, c := make([]float32, m*k), make([]float32, k*n), make([]float32, m*n)
		for i := range a {
			a[i] = float32(i%11-5) * .125
		}
		for i := range b {
			b[i] = float32(i%7-3) * .25
		}
		want := make([]float32, len(c))
		fmaMatrixScalar(want, a, b, m, n, k)
		if !FMAMatrixF32Checked(c, a, b, m, n, k) {
			t.Fatal("edge rejected", shape)
		}
		for i, v := range c {
			if math.Float32bits(v) != math.Float32bits(want[i]) {
				t.Fatal("edge order", shape, i)
			}
		}
	}
	// Partial overlap must fail even when slice starts differ.
	storage := []float32{1, 2, 3, 4, 5, 6}
	if FMAMatrixF32Checked(storage[1:5], storage[:4], b[:4], 2, 2, 2) {
		t.Fatal("partial alias")
	}
	// Mutable legacy dispatch flags cannot bypass the new ISA probe.
	old := HasSgemmAsm
	HasSgemmAsm = !old
	t.Cleanup(func() { HasSgemmAsm = old })
	if !FMAMatrixF32Checked(dst, a, b, 8, 8, 8) {
		t.Fatal("mutable flag affected admission")
	}
}
