package whisper

import "testing"

func TestLayerNormIntoMatchesLayerNorm(t *testing.T) {
	x := []float32{0.01, -0.25, 1.5, 3.0, -2.0, 0.75, 0.33, -0.9}
	weight := []float32{1.0, 0.5, 1.25, 0.75, 1.1, 0.9, 1.4, 0.6}
	bias := []float32{0.1, -0.2, 0.05, 0.3, -0.4, 0.0, 0.2, -0.1}
	want := layerNorm(x, weight, bias, 1, len(x))
	got := make([]float32, len(x))
	layerNormInto(got, x, weight, bias, len(x))
	for i := range want {
		diff := got[i] - want[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > 1e-6 {
			t.Fatalf("idx %d got %.9f want %.9f diff %.9f", i, got[i], want[i], diff)
		}
	}
}
