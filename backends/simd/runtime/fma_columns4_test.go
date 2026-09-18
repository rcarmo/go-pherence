package simd

import (
	"math"
	"reflect"
	"testing"
)

func TestFMAColumns4OrderAndBounds(t *testing.T) {
	for _, cols := range []int{1, 2, 7, 8, 9, 31, 32} {
		for _, k := range []int{1, 3, 8, 31, 251, 300, 400} {
			x := make([]float32, cols*k)
			w := make([]float32, 4*k)
			for i := range x {
				x[i] = float32(math.Sin(float64(i)*.071)) * 3
			}
			for i := range w {
				w[i] = float32(math.Cos(float64(i) * .113))
			}
			got, want := make([]float32, 4*cols), make([]float32, 4*cols)
			fmaColumns4Scalar(want, x, w)
			if !FMAColumns4F32Checked(got, x, w) {
				t.Fatal("rejected", cols, k)
			}
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("cols%d k%d i%d got%x want%x", cols, k, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
				}
			}
		}
	}
}

func TestFMAColumns4RejectsBeforeWrites(t *testing.T) {
	for _, kind := range []string{"dst", "weight", "x", "nanx", "infw", "aliasx", "aliasw"} {
		dst := []float32{7, 8, 9, 10, 11, 12, 13, 14}
		x := []float32{1, 2, 3, 4}
		w := []float32{1, 2, 3, 4, 5, 6, 7, 8}
		switch kind {
		case "dst":
			dst = dst[:7]
		case "weight":
			w = w[:7]
		case "x":
			x = x[:3]
		case "nanx":
			x[2] = float32(math.NaN())
		case "infw":
			w[1] = float32(math.Inf(1))
		case "aliasx":
			dst = x
		case "aliasw":
			dst = w
		}
		before := append([]float32(nil), dst...)
		if FMAColumns4F32Checked(dst, x, w) || !reflect.DeepEqual(dst, before) {
			t.Fatal(kind)
		}
	}
	if !FMAColumns4F32Checked(nil, nil, nil) || FMAColumns4F32Checked(nil, nil, []float32{1}) {
		t.Fatal("empty")
	}
	x, w, dst := make([]float32, 32*251), make([]float32, 4*251), make([]float32, 4*32)
	if n := testing.AllocsPerRun(10, func() {
		if !FMAColumns4F32Checked(dst, x, w) {
			panic("rejected")
		}
	}); n != 0 {
		t.Fatal("allocations", n)
	}
}
