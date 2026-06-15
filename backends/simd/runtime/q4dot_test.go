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
}
