package kernels

import "testing"

func TestApplyRoPEPartialRotatesOnlyRequestedPairs(t *testing.T) {
	x := []float32{1, 2, 3, 4, 10, 20, 30, 40}
	freqs := []float32{0, 1, 1, 0} // pair0: 90° rotation, pair1: identity
	ApplyRoPEPartial(x, freqs, 0, 2, 4, 1)
	want := []float32{-2, 1, 3, 4, -20, 10, 30, 40}
	for i := range want {
		if x[i] != want[i] {
			t.Fatalf("x[%d]=%g want %g (all=%v)", i, x[i], want[i], x)
		}
	}
}

func TestApplyRoPEClampsHeadsAndRotHalf(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	freqs := []float32{0, 1, 0, 1}
	ApplyRoPEPartial(x, freqs, 0, 99, 4, 99)
	want := []float32{-3, -4, 1, 2}
	for i := range want {
		if x[i] != want[i] {
			t.Fatalf("x[%d]=%g want %g (all=%v)", i, x[i], want[i], x)
		}
	}
}

func TestApplyRoPEInvalidInputsAreNoops(t *testing.T) {
	cases := []struct {
		name string
		fn   func([]float32)
	}{
		{"negative pos", func(x []float32) { ApplyRoPEPartial(x, []float32{1, 0}, -1, 1, 2, 1) }},
		{"nil freqs", func(x []float32) { ApplyRoPEPartial(x, nil, 0, 1, 2, 1) }},
		{"zero heads", func(x []float32) { ApplyRoPEPartial(x, []float32{1, 0}, 0, 0, 2, 1) }},
		{"zero headDim", func(x []float32) { ApplyRoPEPartial(x, []float32{1, 0}, 0, 1, 0, 1) }},
		{"zero rot", func(x []float32) { ApplyRoPEPartial(x, []float32{1, 0}, 0, 1, 2, 0) }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			x := []float32{1, 2}
			tc.fn(x)
			if x[0] != 1 || x[1] != 2 {
				t.Fatalf("mutated x=%v", x)
			}
		})
	}
}
