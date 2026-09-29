package nemotrondiarization

import (
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedOfflineHeadPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	head, err := LoadOfflineHead(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	input := readStackingFixture(t, "testdata/jfk_full_tower_normal.f32.gz", 138*projectedWidth)
	original := append([]float32(nil), input...)
	projected, convolved, upsampled, logits, err := head.forwardStages(input, 138)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range []struct {
		name string
		got  []float32
	}{{"projected", projected}, {"convolved", convolved}, {"upsampled", upsampled}, {"logits", logits}} {
		ref := readStackingFixture(t, "testdata/jfk_full_head_"+item.name+".f32.gz", len(item.got))
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
		t.Logf("head %s max_abs=%g mean_abs=%g outside=%d", item.name, maxAbs, mean, outside)
		if outside != 0 || mean > 2e-6 {
			t.Fatalf("head %s differs from PyTorch", item.name)
		}
	}
	out, err := head.ForwardOffline(input, 138)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(logits) || &out[0] == &input[0] {
		t.Fatal("output shape/ownership")
	}
	for i, value := range input {
		if value != original[i] {
			t.Fatalf("mutated input %d", i)
		}
	}
}

func TestReleasedOfflineTowerHeadComposedPyTorchParity(t *testing.T) {
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
	head, err := LoadOfflineHead(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	input := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*projectedWidth)
	initial := append([]float32(nil), input...)
	hidden, err := tower.ForwardOffline(input, 138)
	if err != nil {
		t.Fatal(err)
	}
	logits, err := head.ForwardOffline(hidden, 138)
	if err != nil {
		t.Fatal(err)
	}
	ref := readStackingFixture(t, "testdata/jfk_full_head_logits.f32.gz", len(logits))
	var maxAbs, sumAbs float64
	var outside int
	for i, value := range logits {
		delta := math.Abs(float64(value - ref[i]))
		maxAbs = math.Max(maxAbs, delta)
		sumAbs += delta
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(ref[i])) {
			outside++
		}
	}
	mean := sumAbs / float64(len(logits))
	t.Logf("composed logits max_abs=%g mean_abs=%g outside=%d", maxAbs, mean, outside)
	if outside != 0 || mean > 1e-5 {
		t.Fatal("composed logits differ from PyTorch")
	}
	for i, value := range input {
		if value != initial[i] {
			t.Fatalf("mutated stacking input %d", i)
		}
	}
}

func TestOfflineHeadRejectsMalformed(t *testing.T) {
	if _, err := LoadOfflineHead(nil); err == nil {
		t.Fatal("accepted nil checkpoint")
	}
	if _, err := (*OfflineHead)(nil).ForwardOffline(make([]float32, projectedWidth), 1); err == nil {
		t.Fatal("accepted nil head")
	}
}
