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

func TestSoftmaxInPlaceRejectsNaNInfSum(t *testing.T) {
	if SoftmaxInPlace([]float32{float32(math.Inf(1)), 1}) {
		t.Fatal("SoftmaxInPlace accepted +Inf input")
	}
	if SoftmaxInPlace([]float32{float32(math.NaN()), 1}) {
		t.Fatal("SoftmaxInPlace accepted NaN input")
	}
}

func TestSoftmaxLastAxisTo(t *testing.T) {
	x := []float32{1, 2, 3, 3, 2, 1, 99}
	out := make([]float32, 7)
	out[6] = 123
	if !SoftmaxLastAxisTo(out[:6], x[:6], 2, 3) {
		t.Fatal("SoftmaxLastAxisTo returned false")
	}
	if x[0] != 1 || x[6] != 99 {
		t.Fatalf("SoftmaxLastAxisTo mutated input: %v", x)
	}
	for r := 0; r < 2; r++ {
		sum := float32(0)
		for _, v := range out[r*3 : (r+1)*3] {
			sum += v
		}
		if math.Abs(float64(sum-1)) > 1e-6 {
			t.Fatalf("row %d sum=%g", r, sum)
		}
	}
	if out[6] != 123 {
		t.Fatal("SoftmaxLastAxisTo mutated tail")
	}
	if SoftmaxLastAxisTo(out[:5], x, 2, 3) || SoftmaxLastAxisTo(out, x[:5], 2, 3) || SoftmaxLastAxisTo(out, x, 0, 3) {
		t.Fatal("SoftmaxLastAxisTo accepted malformed input")
	}
}

func TestSoftmaxRowsInPlace(t *testing.T) {
	x := []float32{1, 2, 3, 3, 2, 1, 123}
	if !SoftmaxRowsInPlace(x[:6], 2, 3) {
		t.Fatal("SoftmaxRowsInPlace returned false")
	}
	for r := 0; r < 2; r++ {
		sum := float32(0)
		for _, v := range x[r*3 : (r+1)*3] {
			sum += v
		}
		if math.Abs(float64(sum-1)) > 1e-6 {
			t.Fatalf("row %d sum=%g", r, sum)
		}
	}
	if x[6] != 123 {
		t.Fatal("SoftmaxRowsInPlace mutated tail")
	}
	if SoftmaxRowsInPlace(x[:5], 2, 3) || SoftmaxRowsInPlace(x, 0, 3) || SoftmaxRowsInPlace(x, 2, 0) {
		t.Fatal("SoftmaxRowsInPlace accepted malformed input")
	}
	maxInt := int(^uint(0) >> 1)
	if SoftmaxRowsInPlace(x, maxInt/2+1, 3) {
		t.Fatal("SoftmaxRowsInPlace accepted overflowing dimensions")
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
