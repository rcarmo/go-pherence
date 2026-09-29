package nemotrondiarization

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedLayer1CompletePyTorchParity(t *testing.T) {
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
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	stacked := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*projectedWidth)
	for _, rows := range []int{16, 138} {
		t.Run(fmt.Sprintf("rows=%d", rows), func(t *testing.T) {
			input := append([]float32(nil), stacked[:rows*projectedWidth]...)
			hidden, err := layer0.ForwardOffline(input, rows)
			if err != nil {
				t.Fatal(err)
			}
			initial := append([]float32(nil), hidden...)
			residual, err := layer1.attention.forwardResidual(hidden, rows)
			if err != nil || len(residual) != len(hidden) || &residual[0] == &hidden[0] {
				t.Fatalf("internal attention residual ownership: %v", err)
			}
			_, publicResidual, err := layer1.attention.ForwardOffline(hidden, rows)
			if err != nil {
				t.Fatal(err)
			}
			for i, value := range residual {
				if value != publicResidual[i] {
					t.Fatalf("internal residual differs at %d", i)
				}
			}
			got, err := layer1.ForwardOffline(hidden, rows)
			if err != nil {
				t.Fatal(err)
			}
			name := "testdata/jfk_layer1_complete.f32.gz"
			if rows == 138 {
				name = "testdata/jfk_full_layer1_complete.f32.gz"
			}
			ref := readStackingFixture(t, name, rows*projectedWidth)
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
			t.Logf("max_abs=%g mean_abs=%g outside=%d", maxAbs, mean, outside)
			if outside != 0 || mean > 2e-5 {
				t.Fatal("layer-1 composition differs from PyTorch")
			}
			if &got[0] == &hidden[0] {
				t.Fatal("output aliases input")
			}
			for i, value := range hidden {
				if value != initial[i] {
					t.Fatalf("mutated layer-0 output %d", i)
				}
			}
			for i, value := range input {
				if value != stacked[i] {
					t.Fatalf("mutated caller input %d", i)
				}
			}
		})
	}
}

func TestLayer1CompleteRejectsMalformed(t *testing.T) {
	if _, err := LoadLayer1Complete(nil); err == nil {
		t.Fatal("accepted nil checkpoint")
	}
	if _, err := (*Layer1Complete)(nil).ForwardOffline(make([]float32, projectedWidth), 1); err == nil {
		t.Fatal("accepted nil model")
	}
	if _, err := (&Layer1Complete{}).ForwardOffline(make([]float32, projectedWidth), 1); err == nil {
		t.Fatal("accepted missing weights")
	}
}
