package kernels

import "testing"

func TestAttentionAllocationChecksBeforeMake(t *testing.T) {
	dot := func(a, b []float32) float32 { return 0 }
	saxpy := func(float32, []float32, []float32) {}
	for _, dims := range [][4]int{{1, int(^uint(0) >> 1), 1, 2}, {1, 2, 1, 4}, {-1, 1, 1, 1}, {1, 3, 2, 1}} {
		if got := GQAAttentionScale(nil, nil, nil, dims[0], dims[1], dims[2], dims[3], 1, dot, saxpy); got != nil {
			t.Fatal(dims, got)
		}
	}
}
