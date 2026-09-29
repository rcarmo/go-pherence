package nemotrondiarization

import (
	"fmt"
	"math"
	"os"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedOfflineAudioTowerPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	tower, err := LoadOfflineAudioTower(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	input := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*projectedWidth)
	original := append([]float32(nil), input...)
	hidden, err := tower.first.ForwardOffline(input, 138)
	if err != nil {
		t.Fatal(err)
	}
	for index, layer := range tower.remaining {
		hidden, err = layer.ForwardOffline(hidden, 138)
		if err != nil {
			t.Fatalf("layer %d: %v", index+1, err)
		}
		switch index + 1 {
		case 7, 15, 23, 30:
			compareTowerFixture(t, fmt.Sprintf("layer%d_complete", index+1), hidden)
		}
	}
	out := make([]float32, len(hidden))
	if !simd.LayerNormLastAxisTo(out, hidden, 138, projectedWidth, tower.finalWeight, tower.finalBias, 1e-5) {
		t.Fatal("final norm rejected")
	}
	compareTowerFixture(t, "tower_normal", out)
	composed, err := tower.ForwardOffline(input, 138)
	if err != nil {
		t.Fatal(err)
	}
	compareTowerFixture(t, "tower_normal", composed)
	if &composed[0] == &input[0] {
		t.Fatal("output aliases input")
	}
	for i, value := range input {
		if value != original[i] {
			t.Fatalf("mutated input %d", i)
		}
	}
}

func compareTowerFixture(t *testing.T, name string, got []float32) {
	t.Helper()
	ref := readStackingFixture(t, "testdata/jfk_full_"+name+".f32.gz", 138*projectedWidth)
	var maxAbs, sumAbs float64
	var outside int
	for i, value := range got {
		delta := math.Abs(float64(value - ref[i]))
		maxAbs = math.Max(maxAbs, delta)
		sumAbs += delta
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(ref[i])) {
			outside++
		}
	}
	mean := sumAbs / float64(len(got))
	t.Logf("%s max_abs=%g mean_abs=%g outside=%d", name, maxAbs, mean, outside)
	meanLimit := 2e-5
	if name == "layer30_complete" {
		// Pre-final-norm accumulation over 31 layers is measured separately.
		// The final-normalised output has its own tighter error distribution.
		meanLimit = 3e-5
	}
	if outside != 0 || mean > meanLimit {
		t.Fatalf("%s differs from PyTorch", name)
	}
}

func TestOfflineAudioTowerRejectsMalformed(t *testing.T) {
	if _, err := LoadOfflineAudioTower(nil); err == nil {
		t.Fatal("accepted nil checkpoint")
	}
	if _, err := (*OfflineAudioTower)(nil).ForwardOffline(make([]float32, projectedWidth), 1); err == nil {
		t.Fatal("accepted nil tower")
	}
}
