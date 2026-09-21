package gpu

import "testing"

func TestConv1DInvalidStrideDoesNotPanic(t *testing.T) {
	out := []float32{11}
	Conv1D(out, []float32{1}, []float32{1}, nil, 1, 1, 1, 1, 0, 0)
	if out[0] != 11 {
		t.Fatal(out)
	}
}
