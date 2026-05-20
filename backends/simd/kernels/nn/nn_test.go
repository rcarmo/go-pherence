package nn

import (
	"math"
	"testing"
)

func closeEnough(a, b, tol float32) bool { return float32(math.Abs(float64(a-b))) <= tol }

func TestSoftmaxLastAxisRows(t *testing.T) {
	shape := []int{2, 3}
	got := SoftmaxLastAxis([]float32{1, 2, 3, 3, 2, 1}, shape)
	if len(got) != 6 {
		t.Fatalf("len=%d", len(got))
	}
	for row := 0; row < 2; row++ {
		sum := float32(0)
		for _, v := range got[row*3 : row*3+3] {
			if v <= 0 || v >= 1 {
				t.Fatalf("row %d bad prob %g in %v", row, v, got)
			}
			sum += v
		}
		if !closeEnough(sum, 1, 1e-5) {
			t.Fatalf("row %d sum=%g", row, sum)
		}
	}
	if got[2] <= got[1] || got[1] <= got[0] {
		t.Fatalf("first row not monotonic: %v", got[:3])
	}
	if got[3] <= got[4] || got[4] <= got[5] {
		t.Fatalf("second row not reverse monotonic: %v", got[3:])
	}
}

func TestLayerNormLastAxisWithAffine(t *testing.T) {
	shape := []int{2, 4}
	data := []float32{1, 2, 3, 4, 4, 5, 6, 7}
	gamma := []float32{1, 2, 3, 4}
	beta := []float32{0.5, 0.25, -0.25, -0.5}
	got := LayerNormLastAxis(data, shape, gamma, beta, 1e-5)
	if len(got) != len(data) {
		t.Fatalf("len=%d", len(got))
	}
	for row := 0; row < 2; row++ {
		off := row * 4
		mean := float32(0)
		for _, v := range data[off : off+4] {
			mean += v
		}
		mean /= 4
		var variance float32
		for _, v := range data[off : off+4] {
			d := v - mean
			variance += d * d
		}
		variance /= 4
		inv := float32(1 / math.Sqrt(float64(variance+1e-5)))
		for i := 0; i < 4; i++ {
			want := gamma[i]*(data[off+i]-mean)*inv + beta[i]
			if !closeEnough(got[off+i], want, 1e-5) {
				t.Fatalf("row=%d col=%d got=%g want=%g", row, i, got[off+i], want)
			}
		}
	}
}

func TestGELUApproximationShapeAndBounds(t *testing.T) {
	in := []float32{-4, -1, 0, 1, 4}
	got := GELU(in)
	if len(got) != len(in) {
		t.Fatalf("len=%d", len(got))
	}
	if got[2] != 0 {
		t.Fatalf("GELU(0)=%g", got[2])
	}
	if got[4] <= 3.9 || got[0] > 0 {
		t.Fatalf("unexpected tails: %v", got)
	}
	if got[3] <= 0.8 || got[1] >= 0 {
		t.Fatalf("unexpected middle: %v", got)
	}
}

func TestKernelShapeValidationPanics(t *testing.T) {
	mustPanic := func(name string, fn func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatalf("%s did not panic", name)
			}
		}()
		fn()
	}
	mustPanic("softmax scalar", func() { SoftmaxLastAxis([]float32{1}, nil) })
	mustPanic("softmax bad backing", func() { SoftmaxLastAxis([]float32{1}, []int{2}) })
	mustPanic("layernorm scalar", func() { LayerNormLastAxis([]float32{1}, nil, nil, nil, 1e-5) })
	mustPanic("layernorm affine mismatch", func() { LayerNormLastAxis([]float32{1, 2}, []int{2}, []float32{1}, nil, 1e-5) })
}
