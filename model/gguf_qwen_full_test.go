package model

import "testing"

func TestCopyGGUFQwenFullQ(t *testing.T) {
	qFull := []float32{1, 2, 100, 200, 3, 4, 300, 400}
	dst := make([]float32, 4)
	copyGGUFQwenFullQ(dst, qFull, 2, 2)
	if dst[0] != 1 || dst[1] != 2 || dst[2] != 3 || dst[3] != 4 {
		t.Fatalf("dst=%v", dst)
	}
}

func TestNormGGUFHeads(t *testing.T) {
	x := []float32{3, 4}
	normGGUFHeads(x, []float32{1, 1}, 1, 2, 0)
	if x[0] == 3 || x[1] == 4 {
		t.Fatalf("not normalized: %v", x)
	}
}
