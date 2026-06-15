package simd

import (
	"math"
	"testing"
)

func TestDotI8F32MatchesScalar(t *testing.T) {
	q := make([]byte, 64)
	x := make([]float32, len(q))
	for i := range q {
		q[i] = byte(int8((i*7)%31 - 15))
		x[i] = float32((i%17)-8) * 0.03125
	}
	got, ok := DotI8F32(q, x)
	if !ok {
		t.Fatal("DotI8F32 rejected valid inputs")
	}
	want := dotI8F32Scalar(q, x)
	if math.Abs(float64(got-want)) > 1e-5 {
		t.Fatalf("DotI8F32=%g want %g", got, want)
	}
}

func TestDotI8F32x4MatchesScalar(t *testing.T) {
	q := make([]byte, 64)
	stride := len(q)
	x := make([]float32, 4*stride)
	for i := range q {
		q[i] = byte(int8((i*5)%29 - 14))
	}
	for i := range x {
		x[i] = float32((i%31)-15) * 0.0125
	}
	g0, g1, g2, g3, ok := DotI8F32x4(q, x, stride)
	if !ok {
		t.Fatal("DotI8F32x4 rejected valid inputs")
	}
	w0, w1, w2, w3 := dotI8F32x4Scalar(q, x, stride)
	if math.Abs(float64(g0-w0)) > 1e-5 || math.Abs(float64(g1-w1)) > 1e-5 || math.Abs(float64(g2-w2)) > 1e-5 || math.Abs(float64(g3-w3)) > 1e-5 {
		t.Fatalf("DotI8F32x4=(%g,%g,%g,%g), want (%g,%g,%g,%g)", g0, g1, g2, g3, w0, w1, w2, w3)
	}
}

func TestDotI8F32x4RejectsBadInputs(t *testing.T) {
	if _, _, _, _, ok := DotI8F32x4(nil, nil, 0); ok {
		t.Fatal("expected empty input rejection")
	}
	if _, _, _, _, ok := DotI8F32x4([]byte{1, 2}, make([]float32, 8), 1); ok {
		t.Fatal("expected short stride rejection")
	}
	if _, _, _, _, ok := DotI8F32x4([]byte{1, 2}, make([]float32, 7), 2); ok {
		t.Fatal("expected short x rejection")
	}
}

func TestDotI8F32FallbackHandlesNonMultipleOfEight(t *testing.T) {
	signed := []int8{-3, 2, 7, -11, 5}
	q := make([]byte, len(signed))
	for i, v := range signed {
		q[i] = byte(v)
	}
	x := []float32{0.5, -1.25, 2, -0.75, 0.125}
	got, ok := DotI8F32(q, x)
	if !ok {
		t.Fatal("DotI8F32 rejected non-empty same-length inputs")
	}
	want := dotI8F32Scalar(q, x)
	if got != want {
		t.Fatalf("DotI8F32 fallback=%g want %g", got, want)
	}
}

func TestDotI8F32RejectsBadInputs(t *testing.T) {
	if _, ok := DotI8F32(nil, nil); ok {
		t.Fatal("expected empty input rejection")
	}
	if _, ok := DotI8F32([]byte{1, 2}, []float32{1}); ok {
		t.Fatal("expected short x rejection")
	}
}
