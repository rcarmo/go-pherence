package nemotronasr

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedOfflineProjectionPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	model, err := LoadOfflineProjection(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	features := readStemFixture(t, "features", 32*128)
	original := append([]float32(nil), features...)
	for _, lookahead := range []int{0, 3} {
		tower, encoder, err := model.ForwardFeatures(features, lookahead, 101)
		if err != nil {
			t.Fatal(err)
		}
		for _, item := range []struct {
			name string
			got  []float32
		}{{"tower", tower}, {"encoder", encoder}} {
			ref := readStemFixture(t, fmt.Sprintf("asr_composed_%s_look%d", item.name, lookahead), len(item.got))
			var maxAbs, sumAbs float64
			var outside int
			for i, value := range item.got {
				delta := math.Abs(float64(value - ref[i]))
				maxAbs = math.Max(maxAbs, delta)
				sumAbs += delta
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(ref[i])) {
					outside++
				}
			}
			mean := sumAbs / float64(len(item.got))
			t.Logf("lookahead=%d %s max_abs=%g mean_abs=%g outside=%d", lookahead, item.name, maxAbs, mean, outside)
			if outside != 0 {
				t.Fatalf("lookahead=%d %s differs from PyTorch", lookahead, item.name)
			}
		}
	}
	for i, value := range features {
		if value != original[i] {
			t.Fatalf("mutated features %d", i)
		}
	}
}

func TestOfflineProjectionRejectsMalformed(t *testing.T) {
	if _, err := LoadOfflineProjection(nil); err == nil {
		t.Fatal("accepted nil checkpoint")
	}
	if _, _, err := (*OfflineProjection)(nil).ForwardFeatures(make([]float32, 32*128), 0, 101); err == nil {
		t.Fatal("accepted nil model")
	}
}
