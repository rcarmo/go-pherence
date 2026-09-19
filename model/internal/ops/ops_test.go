package ops

import "testing"

func TestGemvNTInvalidDimensionsAndValidProjection(t *testing.T) {
	for _, dims := range [][2]int{{1, -1}, {-1, 1}, {1, 0}, {int(^uint(0) >> 1), 2}} {
		out := []float32{3, 4}
		GemvNT(out, []float32{1}, []float32{1}, dims[0], dims[1])
		for _, v := range out {
			if v != 0 {
				t.Fatal("invalid call did not clear output", out)
			}
		}
	}
	out := make([]float32, 2)
	GemvNT(out, []float32{1, 2}, []float32{3, 4, 5, 6}, 2, 2)
	if out[0] != 11 || out[1] != 17 {
		t.Fatal(out)
	}
}
