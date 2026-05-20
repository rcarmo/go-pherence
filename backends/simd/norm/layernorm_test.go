package norm

import (
	"math"
	"testing"
)

func TestLayerNormLastAxisTo(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6, 99}
	gamma := []float32{1, 1, 1}
	beta := []float32{0, 0, 0}
	out := make([]float32, len(x))
	out[len(out)-1] = 123
	if !LayerNormLastAxisTo(out[:6], x[:6], 2, 3, gamma, beta, 1e-5) {
		t.Fatal("LayerNormLastAxisTo returned false")
	}
	for r := 0; r < 2; r++ {
		row := out[r*3 : (r+1)*3]
		mean := (row[0] + row[1] + row[2]) / 3
		if math.Abs(float64(mean)) > 1e-5 {
			t.Fatalf("row %d mean=%g", r, mean)
		}
	}
	if out[len(out)-1] != 123 || x[len(x)-1] != 99 {
		t.Fatal("LayerNormLastAxisTo mutated tail/input")
	}
}

func TestLayerNormLastAxisToAllowsInputOutputAliasing(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6}
	want := make([]float32, len(x))
	if !LayerNormLastAxisTo(want, x, 2, 3, nil, nil, 1e-5) {
		t.Fatal("LayerNormLastAxisTo returned false for reference")
	}
	if !LayerNormLastAxisTo(x, x, 2, 3, nil, nil, 1e-5) {
		t.Fatal("LayerNormLastAxisTo returned false for aliased input/output")
	}
	for i := range want {
		if math.Abs(float64(x[i]-want[i])) > 1e-6 {
			t.Fatalf("x[%d]=%g want %g", i, x[i], want[i])
		}
	}
}

func TestLayerNormLastAxisToRejectsMalformedInputs(t *testing.T) {
	x := make([]float32, 6)
	out := make([]float32, 6)
	if LayerNormLastAxisTo(out[:5], x, 2, 3, nil, nil, 1e-5) {
		t.Fatal("accepted short out")
	}
	if LayerNormLastAxisTo(out, x[:5], 2, 3, nil, nil, 1e-5) {
		t.Fatal("accepted short input")
	}
	if LayerNormLastAxisTo(out, x, 0, 3, nil, nil, 1e-5) || LayerNormLastAxisTo(out, x, 2, 0, nil, nil, 1e-5) {
		t.Fatal("accepted zero dimensions")
	}
	if LayerNormLastAxisTo(out, x, 2, 3, []float32{1, 1, 1}, nil, 1e-5) {
		t.Fatal("accepted one-sided affine")
	}
	if LayerNormLastAxisTo(out, x, 2, 3, []float32{1, 1}, []float32{0, 0, 0}, 1e-5) {
		t.Fatal("accepted short gamma")
	}
	maxInt := int(^uint(0) >> 1)
	if LayerNormLastAxisTo(out, x, maxInt/2+1, 3, nil, nil, 1e-5) {
		t.Fatal("accepted overflowing dimensions")
	}
}
