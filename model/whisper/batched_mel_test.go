package whisper

import (
	"math"
	"testing"
)

func TestComputeMelFlatWithTUsesRealMel(t *testing.T) {
	cfg := LargeV3Turbo()
	melCfg := melConfig(cfg)
	samples := make([]float32, 16000)
	for i := range samples {
		samples[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / 16000))
	}
	melFlat, T := computeMelFlatWithT(samples, cfg.NumMelBins, melCfg)
	if T <= 0 {
		t.Fatalf("T=%d", T)
	}
	if len(melFlat) != cfg.NumMelBins*T {
		t.Fatalf("mel len=%d want %d", len(melFlat), cfg.NumMelBins*T)
	}
	var nonZero bool
	for _, v := range melFlat {
		if v != 0 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Fatal("batched mel path returned all-zero placeholder features")
	}
}
