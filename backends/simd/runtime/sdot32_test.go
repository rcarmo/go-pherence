package simd

import (
	"math"
	"math/rand"
	"testing"
)

func TestSdot32MatchesExplicitFMATree(t *testing.T) {
	rng := rand.New(rand.NewSource(32))
	for _, n := range []int{0, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 127, 128, 129, 192, 256, 387, 384, 4097} {
		for trial := 0; trial < 8; trial++ {
			x, y := make([]float32, n+1), make([]float32, n+1)
			for i := range x {
				x[i] = float32(math.Ldexp(rng.Float64()*2-1, rng.Intn(16)-8))
				y[i] = float32(math.Ldexp(rng.Float64()*2-1, rng.Intn(16)-8))
			}
			x, y = x[1:], y[1:]
			want := sdot32Scalar(x, y)
			got := Sdot32(x, y)
			if math.Float32bits(got) != math.Float32bits(want) {
				t.Fatalf("n=%d trial=%d got=%g want=%g", n, trial, got, want)
			}
		}
	}
	defer func() {
		if recover() == nil {
			t.Fatal("mismatched geometry accepted")
		}
	}()
	Sdot32([]float32{1}, nil)
}

func TestSdot32KnownReductionAndAlias(t *testing.T) {
	x := make([]float32, 129)
	for i := range x {
		x[i] = float32(i%5 - 2)
	}
	if got := Sdot32(x, x); got != 256 {
		t.Fatal("integer sum of squares", got)
	}
	// FMA midpoint case: the tiny addend chooses the upper neighbour.
	// Widened F64 FMA followed by F32 narrowing loses that distinction.
	a := float32(1) + float32(math.Ldexp(1, -12))
	b := float32(1) + float32(math.Ldexp(1, -12))
	c := float32(math.Ldexp(1, -80))
	x, y := make([]float32, 64), make([]float32, 64)
	x[0], y[0] = 1, c
	x[32], y[32] = a, b
	want := math.Nextafter32(float32(float64(a)*float64(b)), float32(math.Inf(1)))
	if got := Sdot32(x, y); math.Float32bits(got) != math.Float32bits(want) {
		t.Fatal("FMA midpoint", got, want)
	}
	x[3] = float32(math.NaN())
	if !math.IsNaN(float64(Sdot32(x, x))) {
		t.Fatal("NaN concealed")
	}
}

func TestSdot32ScalarMidpointAndFallback(t *testing.T) {
	a := float32(1) + float32(math.Ldexp(1, -12))
	c := float32(math.Ldexp(1, -80))
	x, y := make([]float32, 64), make([]float32, 64)
	x[0], y[0] = 1, c
	x[32], y[32] = a, a
	want := math.Nextafter32(float32(float64(a)*float64(a)), float32(math.Inf(1)))
	if got := sdot32Scalar(x, y); math.Float32bits(got) != math.Float32bits(want) {
		t.Fatal("scalar midpoint", got, want)
	}
	old := HasDotAsm
	HasDotAsm = false
	t.Cleanup(func() { HasDotAsm = old })
	if got := Sdot32(x, y); math.Float32bits(got) != math.Float32bits(want) {
		t.Fatal("dispatch fallback", got, want)
	}
}
