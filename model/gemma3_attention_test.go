package model

import (
	"math"
	"testing"
)

// This is a CPU-only graph gate for the two Gemma3 attention modes. It keeps
// the llama.cpp ordering explicit: select the SWA/full KV range, scale QK, then
// softmax and multiply by V.
func gemma3ScalarAttentionOracle(q float32, keys, values []float32, scale float32) float32 {
	scores := make([]float64, len(keys))
	maxScore := math.Inf(-1)
	for i := range keys {
		scores[i] = float64(q * keys[i] * scale)
		if scores[i] > maxScore {
			maxScore = scores[i]
		}
	}
	var weighted, sum float64
	for i := range scores {
		w := math.Exp(scores[i] - maxScore)
		sum += w
		weighted += w * float64(values[i])
	}
	return float32(weighted / sum)
}

func TestGemma3SWAAndGlobalAttentionScaleOrdering(t *testing.T) {
	cfg := LlamaConfig{
		ModelType:  "gemma3_text",
		HiddenSize: 4,
		NumHeads:   1,
		NumLayers:  62,
	}
	scale := attentionScale(cfg, 1)
	if scale != 0.5 {
		t.Fatalf("27B synthetic scale=%g want 0.5", scale)
	}

	q := []float32{1}
	k := []float32{-4, -2, 0, 2}
	v := []float32{1, 2, 4, 8}
	cases := []struct {
		name  string
		start int
	}{
		{name: "global", start: 0},
		{name: "sliding", start: 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			keys := k[tc.start:]
			vals := v[tc.start:]
			got := gqaAttentionScale(q, keys, vals, len(keys), 1, 1, 1, scale)
			want := gemma3ScalarAttentionOracle(q[0], keys, vals, scale)
			if len(got) != 1 || math.Abs(float64(got[0]-want)) > 1e-6 {
				t.Fatalf("attention=%v want [%g]", got, want)
			}
			wrong := gemma3ScalarAttentionOracle(q[0], keys, vals, 1)
			if math.Abs(float64(got[0]-wrong)) < 1e-3 {
				t.Fatalf("test vector does not distinguish pre-softmax scaling: got=%g unscaled=%g", got[0], wrong)
			}
		})
	}
}
