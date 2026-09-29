package nemotrondiarization

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedSpeakerCompressionPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	model, err := LoadSpeakerCompressor(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	rows := 486
	embeds := make([]float32, rows*projectedWidth)
	for i := range embeds {
		embeds[i] = float32(i) / 10000
	}
	initialEmbeds := append([]float32(nil), embeds...)
	for _, label := range []string{"pattern", "sweep"} {
		probs := readStackingFixture(t, fmt.Sprintf("testdata/cache_compress_%s_probs.f32.gz", label), rows*diarizationSpeakers)
		initialProbs := append([]float32(nil), probs...)
		got, selected, err := model.Compress(embeds, probs)
		if err != nil {
			t.Fatal(err)
		}
		for _, item := range []struct {
			name string
			got  []float32
		}{{"embeds", got}, {"selected_probs", selected}} {
			ref := readStackingFixture(t, fmt.Sprintf("testdata/cache_compress_%s_%s.f32.gz", label, item.name), len(item.got))
			var maxAbs float64
			var outside int
			for i, value := range item.got {
				delta := math.Abs(float64(value - ref[i]))
				maxAbs = math.Max(maxAbs, delta)
				if math.IsNaN(float64(value)) || delta > 2e-6 {
					outside++
				}
			}
			t.Logf("%s %s max_abs=%g outside=%d", label, item.name, maxAbs, outside)
			if outside != 0 {
				t.Fatalf("%s %s differs from PyTorch", label, item.name)
			}
		}
		if !reflect.DeepEqual(embeds, initialEmbeds) || !reflect.DeepEqual(probs, initialProbs) {
			t.Fatal("mutated compressor input")
		}
	}
}

func TestSpeakerCompressorRejectsMalformed(t *testing.T) {
	if _, err := LoadSpeakerCompressor(nil); err == nil {
		t.Fatal("accepted nil checkpoint")
	}
	if _, _, err := (*SpeakerCompressor)(nil).Compress(make([]float32, 265*projectedWidth), make([]float32, 265*diarizationSpeakers)); err == nil {
		t.Fatal("accepted nil compressor")
	}
}
