package simd

import (
	"math"
	"reflect"
	"testing"
)

func TestFMAColumnsOrderAndBounds(t *testing.T) {
	for _, cols := range []int{0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33} {
		for _, k := range []int{0, 1, 3, 8, 31, 251, 300, 400} {
			for offset := 0; offset < 8; offset++ {
				x := make([]float32, cols*k)
				w := make([]float32, k)
				for i := range x {
					x[i] = float32(math.Sin(float64(i)*.071)) * 3
				}
				for i := range w {
					w[i] = float32(math.Cos(float64(i) * .113))
				}
				all := make([]float32, cols+offset+2)
				for i := range all {
					all[i] = 12345
				}
				dst := all[offset+1 : offset+1+cols]
				want := make([]float32, cols)
				fmaColumnsScalar(want, x, w)
				if !FMAColumnsF32Checked(dst, x, w) {
					t.Fatal("rejected", cols, k, offset)
				}
				for i, v := range dst {
					if math.Float32bits(v) != math.Float32bits(want[i]) {
						t.Fatalf("cols%d k%d offset%d i%d got%x want%x", cols, k, offset, i, math.Float32bits(v), math.Float32bits(want[i]))
					}
				}
				for i, v := range all {
					if (i < offset+1 || i >= offset+1+cols) && v != 12345 {
						t.Fatal("canary")
					}
				}
			}
		}
	}
}
func TestFMAColumnsInvalidAndEmpty(t *testing.T) {
	for _, kind := range []string{"shape", "nanx", "infw", "aliasx", "aliasw", "partial"} {
		dst := []float32{7, 8, 9}
		x := []float32{1, 2, 3, 4, 5, 6}
		w := []float32{1, 2}
		switch kind {
		case "shape":
			x = x[:5]
		case "nanx":
			x[2] = float32(math.NaN())
		case "infw":
			w[1] = float32(math.Inf(1))
		case "aliasx":
			dst = x[:3]
		case "aliasw":
			w = x[:2]
			dst = x[:3]
		case "partial":
			dst = x[1:4]
		}
		before := append([]float32(nil), dst...)
		if FMAColumnsF32Checked(dst, x, w) || !reflect.DeepEqual(dst, before) {
			t.Fatal(kind)
		}
	}
	if FMAColumnsF32Checked(nil, []float32{1}, nil) {
		t.Fatal("empty extent")
	}
	if !FMAColumnsF32Checked(nil, nil, nil) {
		t.Fatal("empty")
	}
	if FMAColumnsF32Checked(nil, nil, []float32{float32(math.NaN())}) {
		t.Fatal("empty nonfinite weight")
	}
	x, w, dst := make([]float32, 32*251), make([]float32, 251), make([]float32, 32)
	if n := testing.AllocsPerRun(10, func() {
		if !FMAColumnsF32Checked(dst, x, w) {
			panic("rejected")
		}
	}); n != 0 {
		t.Fatal("allocations", n)
	}
}
