package whisper

import (
	"math"
	"testing"
)

func TestOriginalDecoderCompatibilityHelpers(t *testing.T) {
	for encLen, want := range map[int]int{0: 0, 1: 255, 256: 0, 257: 255, 1500: 36, 1536: 0} {
		if got := originalCrossPadKeys(encLen); got != want {
			t.Fatal(encLen, got, want)
		}
	}
	x := []float32{1, -2, 0.5}
	ref := append([]float32(nil), x...)
	softmaxPadded(x, 0)
	softmax(ref)
	for i := range x {
		if math.Abs(float64(x[i]-ref[i])) > 1e-7 {
			t.Fatal("pad0", x, ref)
		}
	}
	// Two real logits plus two zero logits: exact denominator e^a+e^b+2.
	y := []float32{1, -1}
	softmaxPadded(y, 2)
	den := math.Exp(1) + math.Exp(-1) + 2
	if math.Abs(float64(y[0])-math.Exp(1)/den) > 1e-6 || math.Abs(float64(y[1])-math.Exp(-1)/den) > 1e-6 {
		t.Fatal("padded", y)
	}
	// All-negative logits: the zero logits set the maximum.
	z := []float32{-50, -60}
	softmaxPadded(z, 36)
	if z[0] <= 0 || z[0] > 1e-20 || math.IsNaN(float64(z[1])) {
		t.Fatal("negative", z)
	}
	g := []float32{-3, -1, 0, 0.5, 2, 10}
	geluOriginalTanh(g)
	for i, v := range []float64{-3, -1, 0, 0.5, 2, 10} {
		want := 0.5 * v * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(v+0.044715*v*v*v)))
		if math.Abs(float64(g[i])-want) > 2e-6*math.Max(1, math.Abs(want)) {
			t.Fatal("gelu", v, g[i], want)
		}
	}
	s := &DecoderState{}
	if s.crossPadKeys != 0 || s.tanhGELU {
		t.Fatal("default state changed")
	}
	s.applyOriginalDecoderCompatibility(1500)
	if s.crossPadKeys != 36 || !s.tanhGELU {
		t.Fatal("compat state")
	}
	if (PCMTranscribeOptions{}).OriginalDecoderCompatibility {
		t.Fatal("default option")
	}
}
