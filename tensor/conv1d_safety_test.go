package tensor

import "testing"

func TestConv1DRejectsRaggedAndOverflow(t *testing.T) {
	for _, c := range []struct {
		x [][]float32
		w [][][]float32
		p int
	}{
		{[][]float32{{1, 2}, {1}}, [][][]float32{{{1}, {1}}}, 0},
		{[][]float32{{1, 2}}, [][][]float32{{{1}}, {{}}}, 0},
		{[][]float32{{1, 2}}, [][][]float32{{{1}}}, int(^uint(0) >> 1)},
		{[][]float32{{1}}, [][][]float32{{{1, 1}}}, 0},
	} {
		if got := Conv1D(c.x, c.w, nil, 1, c.p); got != nil {
			t.Fatal(got)
		}
	}
	for _, dims := range [][6]int{{1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 1, -1}, {1, 1, 1, 1, 1, int(^uint(0) >> 1)}, {int(^uint(0) >> 1), 2, 1, 1, 1, 0}, {1, 1, 1, 2, 2, 0}} {
		out := []float32{9}
		Conv1DFlat(out, []float32{1}, []float32{1}, nil, dims[0], dims[1], dims[2], dims[3], dims[4], dims[5])
		if out[0] != 9 {
			t.Fatal(dims, out)
		}
	}
}
func TestConv1DFlatPaddingOracleAndTail(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	w := []float32{1, 2, 1}
	out := []float32{99, 99, 99, 99, 99}
	Conv1DFlat(out, x, w, []float32{.5}, 1, 4, 1, 3, 1, 1)
	want := []float32{4.5, 8.5, 12.5, 11.5, 99}
	for i := range want {
		if out[i] != want[i] {
			t.Fatal(out, want)
		}
	}
}
