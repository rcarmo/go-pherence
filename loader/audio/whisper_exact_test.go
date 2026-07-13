package audio

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

type whisperLogMelFixture struct {
	Transformers string    `json:"transformers"`
	Shape        []int     `json:"shape"`
	Samples      []float32 `json:"samples"`
	Features     []float32 `json:"features"`
}

func TestWhisperLogMel80TransformersFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/whisper_logmel_transformers_4_57_1.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture whisperLogMelFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	got, frames := WhisperLogMel80(fixture.Samples)
	if len(fixture.Shape) != 2 || fixture.Shape[0] != whisperMels || fixture.Shape[1] != frames || len(got) != len(fixture.Features) {
		t.Fatalf("shape got [%d,%d] fixture=%v values=%d/%d", whisperMels, frames, fixture.Shape, len(got), len(fixture.Features))
	}
	var maxDiff float64
	for i, want := range fixture.Features {
		diff := math.Abs(float64(got[i] - want))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1e-5 {
			t.Fatalf("Transformers %s feature[%d]=%.9g want %.9g diff %.3g", fixture.Transformers, i, got[i], want, diff)
		}
	}
	t.Logf("Transformers %s max abs diff %.3g", fixture.Transformers, maxDiff)
}

func TestWhisperLogMel80Silence(t *testing.T) {
	got, frames := WhisperLogMel80(make([]float32, 1600))
	if frames != 10 || len(got) != 80*10 {
		t.Fatalf("shape=[80,%d] values=%d", frames, len(got))
	}
	for i, value := range got {
		if value != -1.5 {
			t.Fatalf("silence[%d]=%v want -1.5", i, value)
		}
	}
}

func TestWhisperSlaneyEndpoints(t *testing.T) {
	for _, hz := range []float64{0, 100, 999, 1000, 2000, 8000} {
		got := slaneyMelToHz(slaneyHzToMel(hz))
		if math.Abs(got-hz) > 1e-10*math.Max(1, hz) {
			t.Fatalf("Slaney round trip %v -> %v", hz, got)
		}
	}
}
