package simd

import (
	"math"
	"testing"
)

func TestDotU4F32LowAndSumMatchesScalar(t *testing.T) {
	q := make([]byte, 64)
	x := make([]float32, len(q))
	for i := range q {
		q[i] = byte((i*17 + 9) & 0xff)
		x[i] = float32((i%23)-11) * 0.015625
	}
	gotDot, gotSum, ok := DotU4F32LowAndSum(q, x)
	if !ok {
		t.Fatal("DotU4F32LowAndSum rejected valid inputs")
	}
	wantDot, wantSum := dotU4F32LowAndSumScalar(q, x)
	if math.Abs(float64(gotDot-wantDot)) > 1e-5 || math.Abs(float64(gotSum-wantSum)) > 1e-5 {
		t.Fatalf("low dot/sum=(%g,%g), want (%g,%g)", gotDot, gotSum, wantDot, wantSum)
	}
}

func TestDotU4F32HighAndSumMatchesScalar(t *testing.T) {
	q := make([]byte, 64)
	x := make([]float32, len(q))
	for i := range q {
		q[i] = byte((i*29 + 3) & 0xff)
		x[i] = float32((i%19)-9) * 0.03125
	}
	gotDot, gotSum, ok := DotU4F32HighAndSum(q, x)
	if !ok {
		t.Fatal("DotU4F32HighAndSum rejected valid inputs")
	}
	wantDot, wantSum := dotU4F32HighAndSumScalar(q, x)
	if math.Abs(float64(gotDot-wantDot)) > 1e-5 || math.Abs(float64(gotSum-wantSum)) > 1e-5 {
		t.Fatalf("high dot/sum=(%g,%g), want (%g,%g)", gotDot, gotSum, wantDot, wantSum)
	}
}

func TestDotU4F32LowAndSumx4MatchesScalar(t *testing.T) {
	q := make([]byte, 64)
	stride := len(q)
	x := make([]float32, 4*stride)
	for i := range q {
		q[i] = byte((i*17 + 9) & 0xff)
	}
	for i := range x {
		x[i] = float32((i%23)-11) * 0.015625
	}
	g0, gs0, g1, gs1, g2, gs2, g3, gs3, ok := DotU4F32LowAndSumx4(q, x, stride)
	if !ok {
		t.Fatal("DotU4F32LowAndSumx4 rejected valid inputs")
	}
	w0, ws0, w1, ws1, w2, ws2, w3, ws3 := dotU4F32LowAndSumx4Scalar(q, x, stride)
	if math.Abs(float64(g0-w0)) > 1e-5 || math.Abs(float64(gs0-ws0)) > 1e-5 ||
		math.Abs(float64(g1-w1)) > 1e-5 || math.Abs(float64(gs1-ws1)) > 1e-5 ||
		math.Abs(float64(g2-w2)) > 1e-5 || math.Abs(float64(gs2-ws2)) > 1e-5 ||
		math.Abs(float64(g3-w3)) > 1e-5 || math.Abs(float64(gs3-ws3)) > 1e-5 {
		t.Fatalf("low x4=(%g,%g,%g,%g,%g,%g,%g,%g), want (%g,%g,%g,%g,%g,%g,%g,%g)", g0, gs0, g1, gs1, g2, gs2, g3, gs3, w0, ws0, w1, ws1, w2, ws2, w3, ws3)
	}
}

func TestDotU4F32HighAndSumx4MatchesScalar(t *testing.T) {
	q := make([]byte, 64)
	stride := len(q)
	x := make([]float32, 4*stride)
	for i := range q {
		q[i] = byte((i*29 + 3) & 0xff)
	}
	for i := range x {
		x[i] = float32((i%19)-9) * 0.03125
	}
	g0, gs0, g1, gs1, g2, gs2, g3, gs3, ok := DotU4F32HighAndSumx4(q, x, stride)
	if !ok {
		t.Fatal("DotU4F32HighAndSumx4 rejected valid inputs")
	}
	w0, ws0, w1, ws1, w2, ws2, w3, ws3 := dotU4F32HighAndSumx4Scalar(q, x, stride)
	if math.Abs(float64(g0-w0)) > 1e-5 || math.Abs(float64(gs0-ws0)) > 1e-5 ||
		math.Abs(float64(g1-w1)) > 1e-5 || math.Abs(float64(gs1-ws1)) > 1e-5 ||
		math.Abs(float64(g2-w2)) > 1e-5 || math.Abs(float64(gs2-ws2)) > 1e-5 ||
		math.Abs(float64(g3-w3)) > 1e-5 || math.Abs(float64(gs3-ws3)) > 1e-5 {
		t.Fatalf("high x4=(%g,%g,%g,%g,%g,%g,%g,%g), want (%g,%g,%g,%g,%g,%g,%g,%g)", g0, gs0, g1, gs1, g2, gs2, g3, gs3, w0, ws0, w1, ws1, w2, ws2, w3, ws3)
	}
}

func TestDotU4F32AndSumFallbackHandlesNonMultipleOfEight(t *testing.T) {
	q := []byte{0x10, 0x23, 0x45, 0x67, 0x89}
	x := []float32{0.5, -1.25, 2, -0.75, 0.125}
	gotDot, gotSum, ok := DotU4F32LowAndSum(q, x)
	if !ok {
		t.Fatal("DotU4F32LowAndSum rejected non-empty same-length inputs")
	}
	wantDot, wantSum := dotU4F32LowAndSumScalar(q, x)
	if gotDot != wantDot || gotSum != wantSum {
		t.Fatalf("low fallback=(%g,%g), want (%g,%g)", gotDot, gotSum, wantDot, wantSum)
	}
	gotDot, gotSum, ok = DotU4F32HighAndSum(q, x)
	if !ok {
		t.Fatal("DotU4F32HighAndSum rejected non-empty same-length inputs")
	}
	wantDot, wantSum = dotU4F32HighAndSumScalar(q, x)
	if gotDot != wantDot || gotSum != wantSum {
		t.Fatalf("high fallback=(%g,%g), want (%g,%g)", gotDot, gotSum, wantDot, wantSum)
	}
}

func TestDotU4F32AndSumRejectsBadInputs(t *testing.T) {
	if _, _, ok := DotU4F32LowAndSum(nil, nil); ok {
		t.Fatal("expected low empty input rejection")
	}
	if _, _, ok := DotU4F32HighAndSum([]byte{1, 2}, []float32{1}); ok {
		t.Fatal("expected high short x rejection")
	}
	if _, _, _, _, _, _, _, _, ok := DotU4F32LowAndSumx4(nil, nil, 0); ok {
		t.Fatal("expected low x4 empty input rejection")
	}
	if _, _, _, _, _, _, _, _, ok := DotU4F32HighAndSumx4([]byte{1, 2}, make([]float32, 8), 1); ok {
		t.Fatal("expected high x4 short stride rejection")
	}
	if _, _, _, _, _, _, _, _, ok := DotU4F32LowAndSumx4([]byte{1, 2}, make([]float32, 7), 2); ok {
		t.Fatal("expected low x4 short x rejection")
	}
}
