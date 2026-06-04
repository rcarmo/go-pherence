package hunyuan3d

import (
	"math"
	"testing"
)

func TestLinearFloat32(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6}
	w := []float32{1, 0, -1, 2, 1, 0}
	b := []float32{0.5, -1}
	dst := make([]float32, 4)
	if err := LinearFloat32(dst, x, w, b, 2, 3, 2); err != nil {
		t.Fatal(err)
	}
	want := []float32{-1.5, 3, -1.5, 12}
	assertCloseSlice(t, dst, want, 1e-6)
}

func TestRMSNormFloat32(t *testing.T) {
	x := []float32{3, 4}
	w := []float32{1, 2}
	dst := make([]float32, 2)
	if err := RMSNormFloat32(dst, x, w, 1, 2, 0); err != nil {
		t.Fatal(err)
	}
	rms := float32(math.Sqrt((9 + 16) / 2.0))
	want := []float32{3 / rms, 8 / rms}
	assertCloseSlice(t, dst, want, 1e-5)
}

func TestPatchEmbedFloat32(t *testing.T) {
	// CHW: one channel, 4x4 image, 2x2 patches, two embedding channels.
	img := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	// embed0 sums patch, embed1 picks top-left*2.
	w := []float32{1, 1, 1, 1, 2, 0, 0, 0}
	b := []float32{0, 1}
	dst := make([]float32, 4*2)
	n, err := PatchEmbedFloat32(dst, img, w, b, 1, 4, 4, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if n != 4 {
		t.Fatalf("patches=%d", n)
	}
	want := []float32{14, 3, 22, 7, 46, 19, 54, 23}
	assertCloseSlice(t, dst, want, 1e-6)
}

func TestAttentionFloat32(t *testing.T) {
	// One head, two q tokens, two kv tokens, headDim=2.
	q := []float32{1, 0, 0, 1}
	k := []float32{1, 0, 0, 1}
	v := []float32{10, 0, 0, 20}
	dst := make([]float32, 4)
	if err := AttentionFloat32(dst, q, k, v, 2, 2, 1, 2, 1); err != nil {
		t.Fatal(err)
	}
	a := float32(math.Exp(1) / (math.Exp(1) + 1))
	b := float32(1 / (math.Exp(1) + 1))
	want := []float32{10 * a, 20 * b, 10 * b, 20 * a}
	assertCloseSlice(t, dst, want, 1e-5)
}

func TestGELUTanhInPlace(t *testing.T) {
	x := []float32{-1, 0, 1}
	GELUTanhInPlace(x)
	if !(x[0] < 0 && x[1] == 0 && x[2] > 0.8) {
		t.Fatalf("gelu=%v", x)
	}
}

func assertCloseSlice(t *testing.T, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len got=%d want=%d", len(got), len(want))
	}
	for i := range got {
		if diff := float32(math.Abs(float64(got[i] - want[i]))); diff > tol {
			t.Fatalf("[%d] got=%g want=%g diff=%g full=%v", i, got[i], want[i], diff, got)
		}
	}
}
