package nemotrondiarization

import (
	"context"
	"fmt"
	"math"
	"os"
	"testing"

	ptx "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Bounded mixed-row lifecycle and numerical soak, separate from labelled DER.
// One resident tower is reused for exact-key windows and then closed. Neither
// output equality by hash nor a widened gate substitutes for measured errors.
func TestReleasedPTXAudioTowerMixedRowsSoak(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_PTX_SOAK") != "1" {
		t.Skip("set GO_PHERENCE_TEST_NEMOTRON_PTX_SOAK=1")
	}
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" || !ptx.SgemmReady() {
		t.Skip("set model path and CUDA")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	model, err := LoadOfflineAudioTower(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	stacked := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*projectedWidth)
	rowsList := []int{13, 16, 31, 58, 103, 138, 13, 103}
	previousStats := ptx.SetStatsEnabled(true)
	defer ptx.SetStatsEnabled(previousStats)
	before := ptx.StatsSnapshot()
	for cycle := 0; cycle < 2; cycle++ {
		tower, err := NewPTXAudioTower(model, 138)
		if err != nil {
			t.Fatal(err)
		}
		for _, rows := range rowsList {
			input := stacked[:rows*projectedWidth]
			got, err := tower.ForwardRows(context.Background(), input, rows)
			if err != nil {
				tower.Close()
				t.Fatal(err)
			}
			ref, err := model.ForwardOffline(input, rows)
			if err != nil {
				tower.Close()
				t.Fatal(err)
			}
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
			t.Logf("cycle=%d rows=%d max_abs=%g mean_abs=%g outside=%d", cycle, rows, maxAbs, mean, outside)
			if outside != 0 || mean > 2e-5 {
				tower.Close()
				t.Fatalf("mixed-row PTX tower differs from CPU composition at rows=%d", rows)
			}
		}
		tower.Close()
		tower.Close()
	}
	after := ptx.StatsSnapshot()
	if after.Mallocs-before.Mallocs != after.Frees-before.Frees || after.MallocBytes-before.MallocBytes != after.FreeBytes-before.FreeBytes {
		t.Fatalf("PTX tower leaked device memory: before=%+v after=%+v", before, after)
	}
}

func TestReleasedPTXAudioLayerHighAmplitude(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_PTX_SOAK") != "1" {
		t.Skip("set GO_PHERENCE_TEST_NEMOTRON_PTX_SOAK=1")
	}
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" || !ptx.SgemmReady() {
		t.Skip("set model path and CUDA")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	model, err := LoadLayer1Complete(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	layer, err := NewPTXAudioLayer(model, 138)
	if err != nil {
		t.Fatal(err)
	}
	defer layer.Close()
	for _, rows := range []int{13, 103, 138} {
		t.Run(fmt.Sprintf("rows=%d", rows), func(t *testing.T) {
			input := make([]float32, rows*projectedWidth)
			for i := range input {
				input[i] = float32((i*19)%41-20)*8 + float32(i%7)/8
			}
			before := append([]float32(nil), input...)
			ref, err := model.ForwardOffline(input, rows)
			if err != nil {
				t.Fatal(err)
			}
			got, err := layer.Forward(input, rows)
			if err != nil {
				t.Fatal(err)
			}
			var maxAbs, sumAbs float64
			var outside int
			for i, value := range got {
				if input[i] != before[i] {
					t.Fatalf("input mutated at %d", i)
				}
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
				t.Fatal("high-amplitude PTX layer differs from CPU composition")
			}
		})
	}
}
