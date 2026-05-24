package tensor

import (
	"math"
	"testing"
)

func TestMeanStdPool(t *testing.T) {
	// 2 channels, 4 time steps
	// ch0: [1, 2, 3, 4] → mean=2.5, std=sqrt(1.25)≈1.118
	// ch1: [4, 4, 4, 4] → mean=4, std=0
	h := []float32{1, 2, 3, 4, 4, 4, 4, 4}
	out := MeanStdPool(h, 2, 4)
	if len(out) != 4 {
		t.Fatalf("output length=%d want 4", len(out))
	}
	if math.Abs(float64(out[0])-2.5) > 0.01 {
		t.Fatalf("mean[0]=%f want 2.5", out[0])
	}
	if math.Abs(float64(out[1])-4.0) > 0.01 {
		t.Fatalf("mean[1]=%f want 4.0", out[1])
	}
	expectedStd := float32(math.Sqrt(1.25))
	if math.Abs(float64(out[2]-expectedStd)) > 0.01 {
		t.Fatalf("std[0]=%f want %f", out[2], expectedStd)
	}
	if out[3] != 0 {
		t.Fatalf("std[1]=%f want 0", out[3])
	}
}

func TestAttentiveStatPool(t *testing.T) {
	// Simple: 2 channels, 3 time steps, attnDim=1
	// W=[1,1], b=[0], V=[1], vBias=0
	h := []float32{1, 2, 3, 4, 5, 6} // ch0=[1,2,3], ch1=[4,5,6]
	W := []float32{1, 1}             // attnDim=1, channels=2
	b := []float32{0}
	V := []float32{1}

	out := AttentiveStatPool(h, 2, 3, W, b, V, 0, 1)
	if len(out) != 4 {
		t.Fatalf("output length=%d want 4", len(out))
	}
	// Output should be weighted mean/std (softmax-weighted by attention scores)
	// Just verify non-nil and reasonable values
	if out[0] == 0 && out[1] == 0 {
		t.Fatal("attentive pool returned all zeros")
	}
	t.Logf("AttentiveStatPool: mean=[%f,%f] std=[%f,%f]", out[0], out[1], out[2], out[3])
}

func TestAttentiveStatPoolNil(t *testing.T) {
	if out := AttentiveStatPool(nil, 0, 0, nil, nil, nil, 0, 0); out != nil {
		t.Fatalf("expected nil for zero dimensions")
	}
}
