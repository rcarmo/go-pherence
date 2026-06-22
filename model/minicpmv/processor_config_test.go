package minicpmv

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/config"
)

func TestImagePreprocessConfigFromProcessor(t *testing.T) {
	proc := config.MiniCPMVProcessorConfig{
		NormalizedSize: 336,
		PatchSize:      14,
		DoResize:       true,
		DoRescale:      true,
		DoNormalize:    true,
		RescaleFactor:  1.0 / 255.0,
		ImageMean:      []float32{0.1, 0.2, 0.3},
		ImageStd:       []float32{0.4, 0.5, 0.6},
	}
	cfg := ImagePreprocessConfigFromProcessor(proc, 448, 16)
	if cfg.Size != 336 || cfg.PatchSize != 14 || !cfg.DoResize || cfg.ImageMean != [3]float32{0.1, 0.2, 0.3} || cfg.ImageStd != [3]float32{0.4, 0.5, 0.6} {
		t.Fatalf("bad preprocess config: %+v", cfg)
	}
}
