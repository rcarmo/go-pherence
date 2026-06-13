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
