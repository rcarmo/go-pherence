package audio

import (
	"math"
	"testing"
)

func TestResampleInvalidRatesAndBudgets(t *testing.T) {
	for _, fn := range []func([]float32, int, int) []float32{Resample, ResampleSinc} {
		for _, r := range [][2]int{{0, 1}, {1, 0}, {-1, 16000}, {0, 0}, {1, int(^uint(0) >> 1)}} {
			if got := fn([]float32{1, 2}, r[0], r[1]); got != nil {
				t.Fatal(r, len(got))
			}
		}
	}
	if _, ok := resampleLength(int(^uint(0)>>1), 1, 2); ok {
		t.Fatal("overflow")
	}
	if _, ok := resampleLength(maxResampleSamples+1, 1, 1); ok {
		t.Fatal("budget")
	}
}
func TestResampleLengthExactAndConstantSignal(t *testing.T) {
	for _, c := range [][4]int{{3, 3, 2, 2}, {441, 44100, 16000, 160}, {2, 16000, 48000, 6}} {
		n, ok := resampleLength(c[0], c[1], c[2])
		if !ok || n != c[3] {
			t.Fatal(c, n, ok)
		}
	}
	src := make([]float32, 31)
	for i := range src {
		src[i] = .5
	}
	for _, fn := range []func([]float32, int, int) []float32{Resample, ResampleSinc} {
		out := fn(src, 16000, 48000)
		if len(out) != 93 {
			t.Fatal(len(out))
		}
		for _, v := range out {
			if math.IsNaN(float64(v)) || math.Abs(float64(v-.5)) > 1e-6 {
				t.Fatal(v)
			}
		}
	}
}
