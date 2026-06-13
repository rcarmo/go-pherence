package diffusiongemma

import (
	"math"
	"math/rand"
	"testing"
)

func TestTokenStatsFromLogitsMatchesSeparateHelpers(t *testing.T) {
	cases := [][]float32{
		{1, 3, 2, -1},
		{float32(math.Inf(-1)), 9, float32(math.Inf(-1)), 4},
		{0, 0, 0, 0},
	}
	for _, logits := range cases {
		temperature := 0.7
		scaled := ApplyTemperature(logits, temperature)
		wantArgmax := Argmax(scaled)
		wantEntropy := TokenEntropyFromLogits(scaled)
		wantSample := SampleFromLogits(scaled, rand.New(rand.NewSource(42)))
		gotArgmax, gotSample, gotEntropy := TokenStatsFromLogits(logits, temperature, rand.New(rand.NewSource(42)))
		if gotArgmax != wantArgmax || gotSample != wantSample || math.Abs(gotEntropy-wantEntropy) > 1e-9 {
			t.Fatalf("TokenStats mismatch logits=%v got=(%d,%d,%g) want=(%d,%d,%g)", logits, gotArgmax, gotSample, gotEntropy, wantArgmax, wantSample, wantEntropy)
		}
	}
}

func TestArgmaxEntropyFromLogitsMatchesSeparateHelpers(t *testing.T) {
	cases := [][]float32{
		{1, 3, 2, -1},
		{float32(math.Inf(-1)), 9, float32(math.Inf(-1)), 4},
		{0, 0, 0, 0},
	}
	for _, logits := range cases {
		temperature := 0.7
		scaled := ApplyTemperature(logits, temperature)
		wantArgmax := Argmax(scaled)
		wantEntropy := TokenEntropyFromLogits(scaled)
		rng := rand.New(rand.NewSource(42))
		gotArgmax, gotEntropy := ArgmaxEntropyFromLogits(logits, temperature, rng)
		if gotArgmax != wantArgmax || math.Abs(gotEntropy-wantEntropy) > 1e-9 {
			t.Fatalf("ArgmaxEntropy mismatch logits=%v got=(%d,%g) want=(%d,%g)", logits, gotArgmax, gotEntropy, wantArgmax, wantEntropy)
		}
		wantNext := rand.New(rand.NewSource(42))
		_ = wantNext.Float64()
		if got, want := rng.Float64(), wantNext.Float64(); got != want {
			t.Fatalf("ArgmaxEntropy did not preserve RNG progression: got next %g want %g", got, want)
		}
	}
}
