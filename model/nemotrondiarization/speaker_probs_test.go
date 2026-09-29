package nemotrondiarization

import (
	"fmt"
	"math"
	"reflect"
	"testing"
)

func TestSpeakerProbabilitiesPyTorchParity(t *testing.T) {
	for index, rows := range []int{13, 22, 31} {
		logits := make([]float32, rows*diarizationUpsample*diarizationSpeakers)
		for i := range logits {
			logits[i] = float32(i%53-26) / 11
		}
		mask := make([]bool, rows)
		for i := range mask {
			mask[i] = index != 2 || i != rows-1
		}
		initial := append([]float32(nil), logits...)
		got, err := PoolSpeakerProbabilities(logits, mask)
		if err != nil {
			t.Fatal(err)
		}
		ref := readStackingFixture(t, fmt.Sprintf("testdata/cache_step%d_pooled.f32.gz", index), rows*diarizationSpeakers)
		var maxAbs, sumAbs float64
		var outside int
		for i, value := range got {
			delta := math.Abs(float64(value - ref[i]))
			maxAbs = math.Max(maxAbs, delta)
			sumAbs += delta
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 2e-6 {
				outside++
			}
		}
		t.Logf("step=%d max_abs=%g mean_abs=%g outside=%d", index, maxAbs, sumAbs/float64(len(got)), outside)
		if outside != 0 {
			t.Fatalf("step=%d probabilities differ from PyTorch", index)
		}
		if !reflect.DeepEqual(logits, initial) {
			t.Fatal("mutated logits")
		}
	}
}

func TestSpeakerProbabilitiesRejectsMalformed(t *testing.T) {
	if _, err := PoolSpeakerProbabilities(make([]float32, 7), []bool{true}); err == nil {
		t.Fatal("accepted short logits")
	}
	bad := make([]float32, 8*diarizationSpeakers)
	bad[0] = float32(math.NaN())
	if _, err := PoolSpeakerProbabilities(bad, []bool{false}); err == nil {
		t.Fatal("accepted non-finite masked logits")
	}
}
