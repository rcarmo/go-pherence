package speaker

import (
	"math"
	"testing"
)

func TestBatchNormForwardEval(t *testing.T) {
	bn := BatchNorm1D{Weight: []float32{2}, Bias: []float32{1}, RunningMean: []float32{3}, RunningVar: []float32{4}}
	out := batchNormForward(bn, []float32{5, 7}, 1, 2)
	if len(out) != 2 || math.Abs(float64(out[0]-3)) > 1e-4 || math.Abs(float64(out[1]-5)) > 1e-4 {
		t.Fatalf("bn=%v", out)
	}
}

func TestConv1DForwardDilatedSameLength(t *testing.T) {
	conv := Conv1D{Weight: []float32{1, 10, 100}, Shape: []int{1, 1, 3}}
	out := conv1dForward(conv, []float32{1, 2, 3, 4}, 1, 4, 2)
	want := []float32{310, 420, 31, 42}
	if len(out) != len(want) {
		t.Fatalf("len=%d", len(out))
	}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("out=%v want %v", out, want)
		}
	}
}

func TestSoftmaxTimePerChannel(t *testing.T) {
	x := []float32{1, 2, 3, 4, 4, 4}
	softmaxTimePerChannel(x, 2, 3)
	for c := 0; c < 2; c++ {
		var sum float32
		for t := 0; t < 3; t++ {
			sum += x[c*3+t]
		}
		if math.Abs(float64(sum-1)) > 1e-5 {
			t.Fatalf("channel %d sum=%f x=%v", c, sum, x)
		}
	}
}

func TestL2Normalize(t *testing.T) {
	x := l2Normalize([]float32{3, 4})
	if math.Abs(float64(x[0]-0.6)) > 1e-6 || math.Abs(float64(x[1]-0.8)) > 1e-6 {
		t.Fatalf("x=%v", x)
	}
}
