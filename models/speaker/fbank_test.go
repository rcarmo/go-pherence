package speaker

import "testing"

func TestSpeechBrainFbankShapeAndMeanNorm(t *testing.T) {
	samples := make([]float32, 16000)
	for i := range samples {
		samples[i] = 0.1
	}
	f := SpeechBrainFbank(samples, 16000)
	if len(f) != 80 || len(f[0]) == 0 {
		t.Fatalf("shape=%dx%d", len(f), len(f[0]))
	}
	for m := range f {
		var mean float32
		for _, v := range f[m] {
			mean += v
		}
		mean /= float32(len(f[m]))
		if mean < -1e-4 || mean > 1e-4 {
			t.Fatalf("mel %d mean=%g", m, mean)
		}
	}
}

func TestSpeechBrainFbankRejectsNon16k(t *testing.T) {
	if got := SpeechBrainFbank([]float32{1, 2, 3}, 8000); got != nil {
		t.Fatalf("expected nil for non-16k input")
	}
}
