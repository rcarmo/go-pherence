package qwen

import (
	"math"
	"testing"
)

func TestBranchL2EpsilonPlacement(t *testing.T) {
	for _, input := range [][]float32{{0, 0}, {0.00001, -0.00002}, {3, 4}, {0.3, -0.9, 0.7}} {
		got := append([]float32(nil), input...)
		var sum float32
		for _, v := range input {
			sum += v * v
		}
		inv := float32(1 / math.Sqrt(float64(sum+1e-6)))
		qwen35BranchL2Norm(got, 1e-6)
		for i, v := range got {
			if v != input[i]*inv {
				t.Fatal("inside epsilon", input, got)
			}
		}
	}
	x := []float32{0.00001, -0.00002}
	old := append([]float32(nil), x...)
	qwen35BranchL2Norm(x, 1e-6)
	l2NormalizeInPlace(old, 1e-6)
	if x[0] == old[0] {
		t.Fatal("fixture does not distinguish formulas")
	}
	if err := qwen35BranchL2Heads(make([]float32, 8), 2, 4, 1e-6); err != nil {
		t.Fatal(err)
	}
	for _, dims := range [][2]int{{0, 4}, {2, 0}, {3, 4}, {2, 5}} {
		if err := qwen35BranchL2Heads(make([]float32, 8), dims[0], dims[1], 1e-6); err == nil {
			t.Fatal("invalid heads")
		}
	}
	buf := make([]float32, 128)
	if a := testing.AllocsPerRun(10, func() { qwen35BranchL2Norm(buf, 1e-6) }); a != 0 {
		t.Fatal("allocation", a)
	}
}
