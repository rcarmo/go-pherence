package simd

import (
	"math"
	"testing"
)

func TestSoftmaxInPlace(t *testing.T) {
	x := []float32{1, 2, 3, 123}
	if !SoftmaxInPlace(x[:3]) {
		t.Fatal("SoftmaxInPlace returned false")
	}
	if x[3] != 123 {
		t.Fatalf("SoftmaxInPlace mutated tail: %v", x)
	}
	sum := float32(0)
	for _, v := range x[:3] {
		if v <= 0 || v >= 1 {
			t.Fatalf("softmax value out of range: %v", x[:3])
		}
		sum += v
	}
	if math.Abs(float64(sum-1)) > 1e-6 {
		t.Fatalf("sum=%g want 1", sum)
	}
	if !(x[2] > x[1] && x[1] > x[0]) {
		t.Fatalf("softmax order not preserved: %v", x[:3])
	}
	if SoftmaxInPlace(nil) {
		t.Fatal("SoftmaxInPlace accepted nil input")
	}
}

func TestSoftmaxInPlaceStableLargeValues(t *testing.T) {
	x := []float32{1000, 1001, 1002}
	if !SoftmaxInPlace(x) {
		t.Fatal("SoftmaxInPlace returned false")
	}
	for i, v := range x {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("x[%d]=%v", i, v)
		}
	}
}
