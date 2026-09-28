package pockettts

import (
	"math"
	"math/rand"
	"testing"
)

func TestLSDDecodeSIMDMatchesScalar(t *testing.T) {
	initial := []float32{-.75, -.25, 0, .5, 1}
	network := func(dst []float32, start, target float32, current []float32) error {
		for i := range dst {
			dst[i] = (target-start)*float32(i+1) - .25*current[i]
		}
		return nil
	}
	for _, steps := range []int{1, 2, 4, 16} {
		got, want := make([]float32, len(initial)), make([]float32, len(initial))
		scratchA, scratchB := make([]float32, len(initial)), make([]float32, len(initial))
		if err := LSDDecodeSIMD(got, initial, steps, scratchA, network); err != nil {
			t.Fatal(err)
		}
		if err := lsdDecodeScalar(want, initial, steps, scratchB, network); err != nil {
			t.Fatal(err)
		}
		for i := range got {
			if math.Abs(float64(got[i]-want[i])) > 1e-6 {
				t.Fatalf("steps=%d index=%d got=%g want=%g", steps, i, got[i], want[i])
			}
		}
	}
}

func TestAffineSIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	for _, dims := range [][2]int{{1, 1}, {7, 9}, {32, 64}, {1024, 32}} {
		in, out := dims[0], dims[1]
		x, w, b := make([]float32, in), make([]float32, in*out), make([]float32, out)
		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}
		for i := range w {
			w[i] = rng.Float32()*2 - 1
		}
		for i := range b {
			b[i] = rng.Float32()*2 - 1
		}
		got, want := make([]float32, out), make([]float32, out)
		if err := AffineSIMD(got, x, w, b, in, out); err != nil {
			t.Fatal(err)
		}
		affineScalar(want, x, w, b, in, out)
		for i := range got {
			if math.Abs(float64(got[i]-want[i])) > 2e-4 {
				t.Fatalf("dims=%v index=%d got=%g want=%g", dims, i, got[i], want[i])
			}
		}
	}
}

func TestPocketSIMDPrimitivesRejectMalformed(t *testing.T) {
	if err := LSDDecodeSIMD(nil, nil, 1, nil, func([]float32, float32, float32, []float32) error { return nil }); err == nil {
		t.Fatal("accepted empty LSD")
	}
	if err := AffineSIMD(make([]float32, 1), make([]float32, 2), make([]float32, 1), nil, 2, 1); err == nil {
		t.Fatal("accepted short affine")
	}
	bad := func(dst []float32, _, _ float32, _ []float32) error { dst[0] = float32(math.NaN()); return nil }
	if err := LSDDecodeSIMD(make([]float32, 1), []float32{0}, 1, make([]float32, 1), bad); err == nil {
		t.Fatal("accepted NaN flow")
	}
}
