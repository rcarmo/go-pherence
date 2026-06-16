package diffusiongemma

import (
	"math/rand"
	"testing"
)

func TestQ6KBlockISumMatchesScalar(t *testing.T) {
	q8 := make([]int8, 256)
	var q6 [256]int8
	scales := make([]byte, 16)
	for i := range q8 {
		q8[i] = int8((i*11)%127 - 63)
		q6[i] = int8((i*7)%63 - 31)
	}
	for i := range scales {
		scales[i] = byte(int8((i % 9) - 4))
	}
	got, ok := q6KBlockISum(q8, &q6, scales)
	if !ok {
		t.Fatal("q6KBlockISum rejected valid inputs")
	}
	var want int32
	for group := 0; group < 16; group++ {
		scale := int32(int8(scales[group]))
		base := group * 16
		var acc int32
		for i := 0; i < 16; i++ {
			acc += int32(q8[base+i]) * int32(q6[base+i])
		}
		want += scale * acc
	}
	if got != want {
		t.Fatalf("q6KBlockISum=%d want %d", got, want)
	}
}

func TestQ6KBlockCoeffISumMatchesBlockISum(t *testing.T) {
	q8 := make([]int8, 256)
	var q6 [256]int8
	scales := make([]byte, 16)
	for i := range q8 {
		q8[i] = int8((i*11)%127 - 63)
		q6[i] = int8((i*7)%63 - 31)
	}
	for i := range scales {
		scales[i] = byte(int8((i % 9) - 4))
	}
	want, ok := q6KBlockISum(q8, &q6, scales)
	if !ok {
		t.Fatal("q6KBlockISum rejected valid inputs")
	}
	coeff, ok := q6KBlockScaledCoeffs(&q6, scales)
	if !ok {
		t.Fatal("q6KBlockScaledCoeffs rejected valid inputs")
	}
	got, ok := q6KBlockCoeffISum(q8, &coeff)
	if !ok {
		t.Fatal("q6KBlockCoeffISum rejected valid inputs")
	}
	if got != want {
		t.Fatalf("coeff dot=%d want %d", got, want)
	}
}

func TestQ6I8DotAVX2MatchesScalar(t *testing.T) {
	if !hasQ6KBlockCoeffISumSIMD {
		t.Skip("Q6/I8 AVX2 path is not available on this architecture")
	}
	rng := rand.New(rand.NewSource(0x600d6))
	for trial := 0; trial < 256; trial++ {
		q8 := make([]int8, 256)
		var coeff [256]int16
		for i := 0; i < 256; i++ {
			q8[i] = int8(rng.Intn(256) - 128)
			// Q6 values are [-32,31] and scales are int8, so scaled coeffs
			// fit comfortably in int16. Generate the full legal product range
			// rather than only tiny smoke values.
			q6 := int16(rng.Intn(64) - 32)
			scale := int16(rng.Intn(256) - 128)
			coeff[i] = q6 * scale
		}
		got, ok := q6KBlockCoeffISum(q8, &coeff)
		if !ok {
			t.Fatalf("trial %d: q6KBlockCoeffISum rejected valid inputs", trial)
		}
		want := q6KBlockCoeffISumScalar(q8, &coeff)
		if got != want {
			t.Fatalf("trial %d: AVX2 dot=%d want scalar=%d", trial, got, want)
		}
	}
}

func TestQ6KBlockISumRejectsBadInputs(t *testing.T) {
	var q6 [256]int8
	if _, ok := q6KBlockISum(nil, &q6, make([]byte, 16)); ok {
		t.Fatal("expected q8 rejection")
	}
	if _, ok := q6KBlockISum(make([]int8, 256), nil, make([]byte, 16)); ok {
		t.Fatal("expected nil q6 rejection")
	}
	if _, ok := q6KBlockISum(make([]int8, 256), &q6, make([]byte, 15)); ok {
		t.Fatal("expected scales rejection")
	}
}
