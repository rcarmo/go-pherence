package whisper

import (
	"math"
	"testing"
)

func TestComputeMelFromSamplesUsesRealMel(t *testing.T) {
	cfg := LargeV3Turbo()
	melCfg := melConfigAudio(cfg)
	samples := make([]float32, 16000)
	for i := range samples {
		samples[i] = float32(math.Sin(2 * math.Pi * 880 * float64(i) / 16000))
	}
	melFlat := computeMelFromSamples(samples, cfg.NumMelBins, melCfg)
	if len(melFlat) == 0 {
		t.Fatal("empty mel")
	}
	var nonZero bool
	for _, v := range melFlat {
		if v != 0 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Fatal("language-detect mel path returned all-zero placeholder features")
	}
}
