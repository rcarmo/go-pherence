package kernels

import (
	"math"
	"testing"
)

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

func TestApplyRoPEMatchesReferenceForOddAndTailDims(t *testing.T) {
	cases := []struct {
		name            string
		x               []float32
		pos, heads, dim int
		rotHalf         int
		freqs           []float32
	}{
		{
			name:    "odd head dim leaves center tail untouched",
			x:       []float32{1, 2, 3, 4, 5, 10, 20, 30, 40, 50},
			pos:     1,
			heads:   2,
			dim:     5,
			rotHalf: 2,
			freqs:   []float32{1, 0, 1, 0, 0, 1, -1, 0},
		},
		{
			name:    "partial rotation leaves per-head suffix untouched",
			x:       []float32{1, 2, 3, 4, 5, 6, 99},
			pos:     0,
			heads:   9,
			dim:     6,
			rotHalf: 1,
			freqs:   []float32{0, 1},
		},
		{
			name:    "full rope wrapper matches partial reference",
			x:       []float32{1, 2, 3, 4, 5, 6, 7, 8},
			pos:     0,
			heads:   2,
			dim:     4,
			rotHalf: 2,
			freqs:   []float32{0, 1, 1, 0},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := append([]float32(nil), tc.x...)
			want := append([]float32(nil), tc.x...)
			if tc.rotHalf == tc.dim/2 {
				ApplyRoPE(got, tc.freqs, tc.pos, tc.heads, tc.dim)
			} else {
				ApplyRoPEPartial(got, tc.freqs, tc.pos, tc.heads, tc.dim, tc.rotHalf)
			}
			referenceRoPEPartial(want, tc.freqs, tc.pos, tc.heads, tc.dim, tc.rotHalf)
			assertFloat32SlicesClose(t, got, want, 1e-6)
		})
	}
}

func TestApplyRoPEShortFreqsLeaveRemainingPairsUntouched(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	ApplyRoPEPartial(x, []float32{0, 1}, 0, 1, 4, 2)
	want := []float32{-3, 2, 1, 4}
	assertFloat32SlicesClose(t, x, want, 0)
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

func referenceRoPEPartial(x, freqs []float32, pos, numHeads, headDim, rotHalf int) {
	if pos < 0 || numHeads <= 0 || headDim <= 0 || rotHalf <= 0 || len(x) == 0 || len(freqs) == 0 {
		return
	}
	if rotHalf > headDim/2 {
		rotHalf = headDim / 2
	}
	maxHeads := len(x) / headDim
	if numHeads > maxHeads {
		numHeads = maxHeads
	}
	for h := 0; h < numHeads; h++ {
		base := h * headDim
		for i := 0; i < rotHalf; i++ {
			freqOff := (pos*rotHalf + i) * 2
			if freqOff+1 >= len(freqs) {
				break
			}
			cos, sin := freqs[freqOff], freqs[freqOff+1]
			a, b := base+i, base+i+rotHalf
			x0, x1 := x[a], x[b]
			x[a] = x0*cos - x1*sin
			x[b] = x0*sin + x1*cos
		}
	}
}

func assertFloat32SlicesClose(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("x[%d]=%g want %g (all=%v)", i, got[i], want[i], got)
		}
	}
}
