package nemotrondiarization

import (
	"math"
	"os"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedLayer2CompleteComposedPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	layer0, err := LoadLayer0Complete(file)
	if err != nil {
		t.Fatal(err)
	}
	layer1, err := LoadLayer1Complete(file)
	if err != nil {
		t.Fatal(err)
	}
	layer2, err := LoadIndexedAudioLayer(file, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	input := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*projectedWidth)
	initial := append([]float32(nil), input...)
	hidden, err := layer0.ForwardOffline(input, 138)
	if err != nil {
		t.Fatal(err)
	}
	hidden, err = layer1.ForwardOffline(hidden, 138)
	if err != nil {
		t.Fatal(err)
	}
	prior := append([]float32(nil), hidden...)
	normal := make([]float32, len(hidden))
	if !simd.LayerNormLastAxisTo(normal, hidden, 138, projectedWidth, layer2.attention.qkv.gamma, layer2.attention.qkv.beta, 1e-5) {
		t.Fatal("layer-2 norm rejected")
	}
	compareLayer2Fixture(t, "normal", normal)
	attention, residual, err := layer2.attention.ForwardOffline(hidden, 138)
	if err != nil {
		t.Fatal(err)
	}
	compareLayer2Fixture(t, "attention", attention)
	compareLayer2Fixture(t, "residual", residual)
	output, err := layer2.ForwardOffline(hidden, 138)
	if err != nil {
		t.Fatal(err)
	}
	compareLayer2Fixture(t, "complete", output)
	if &output[0] == &hidden[0] || &attention[0] == &hidden[0] || &residual[0] == &hidden[0] {
		t.Fatal("output aliases input")
	}
	for i, value := range hidden {
		if value != prior[i] {
			t.Fatalf("mutated layer-1 output %d", i)
		}
	}
	for i, value := range input {
		if value != initial[i] {
			t.Fatalf("mutated stacking input %d", i)
		}
	}
}

func compareLayer2Fixture(t *testing.T, name string, got []float32) {
	t.Helper()
	ref := readStackingFixture(t, "testdata/jfk_full_layer2_"+name+".f32.gz", 138*projectedWidth)
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
	t.Logf("layer2 %s max_abs=%g mean_abs=%g outside=%d", name, maxAbs, mean, outside)
	if outside != 0 || mean > 2e-5 {
		t.Fatalf("layer2 %s differs from PyTorch", name)
	}
}

func TestIndexedAudioLayerRejectsInvalidIndex(t *testing.T) {
	for _, layer := range []int{-1, 0, 31, 32} {
		if _, err := LoadIndexedAudioLayer(nil, layer); err == nil {
			t.Fatalf("accepted layer=%d nil checkpoint", layer)
		}
	}
}
