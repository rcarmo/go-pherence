package nemotrondiarization

import (
	"math"
	"testing"
)

func TestSpeakerFrameScoresPyTorchParity(t *testing.T) {
	for _, tc := range []struct {
		name        string
		negativeInf int
	}{{"cache_score", 1200}, {"cache_score_speech", 2280}} {
		probs := readStackingFixture(t, "testdata/"+tc.name+"_probs.f32.gz", 300*diarizationSpeakers)
		initial := append([]float32(nil), probs...)
		got, err := SpeakerFrameScores(probs)
		if err != nil {
			t.Fatal(err)
		}
		ref := readStackingFixture(t, "testdata/"+tc.name+"_output.f32.gz", len(got))
		var maxAbs, sumAbs float64
		var outside, negInf int
		for i, value := range got {
			if math.IsInf(float64(ref[i]), -1) {
				if !math.IsInf(float64(value), -1) {
					outside++
				}
				negInf++
				continue
			}
			delta := math.Abs(float64(value - ref[i]))
			maxAbs = math.Max(maxAbs, delta)
			sumAbs += delta
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 2e-6 {
				outside++
			}
		}
		t.Logf("%s finite max_abs=%g mean_abs=%g negative_inf=%d outside=%d", tc.name, maxAbs, sumAbs/float64(len(got)-negInf), negInf, outside)
		if outside != 0 || negInf != tc.negativeInf {
			t.Fatal("speaker scores differ from PyTorch")
		}
		for i, value := range probs {
			if value != initial[i] {
				t.Fatalf("mutated probabilities %d", i)
			}
		}
	}
}

func TestSpeakerFrameScoresRejectsMalformed(t *testing.T) {
	if _, err := SpeakerFrameScores(make([]float32, 7)); err == nil {
		t.Fatal("accepted short row")
	}
	bad := make([]float32, 8)
	bad[0] = float32(math.NaN())
	if _, err := SpeakerFrameScores(bad); err == nil {
		t.Fatal("accepted non-finite probability")
	}
	bad[0] = 1.1
	if _, err := SpeakerFrameScores(bad); err == nil {
		t.Fatal("accepted probability above one")
	}
}
