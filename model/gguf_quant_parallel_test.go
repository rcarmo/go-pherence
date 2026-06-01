package model

import "testing"

func TestDotF32(t *testing.T) {
	if got := dotF32([]float32{1, 2, 3}, []float32{4, 5, 6}); got != 32 {
		t.Fatalf("dot=%v", got)
	}
}
