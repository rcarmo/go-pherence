package nemotrondiarization

import (
	"context"
	"math"
	"os"
	"testing"

	ptx "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// A full 541-key unmasked window exercises exact attention bounds and the
// maximum scratch capacity. This is CPU-composition parity, not labelled DER.
func TestReleasedPTXAudioTowerMaxPreparedRows(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_PTX_MAX_ROWS") != "1" {
		t.Skip("set GO_PHERENCE_TEST_NEMOTRON_PTX_MAX_ROWS=1")
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
	const rows = maxPreparedDiarizationRows
	if rows != 541 {
		t.Fatalf("unexpected prepared row bound %d", rows)
	}
	stacked := readStackingFixture(t, "testdata/jfk_loop100_stacking_transformers_5_18.f32.gz", 1251*projectedWidth)
	input := append([]float32(nil), stacked[:rows*projectedWidth]...)
	original := append([]float32(nil), input...)
	previous := ptx.SetStatsEnabled(true)
	defer ptx.SetStatsEnabled(previous)
	before := ptx.StatsSnapshot()
	tower, err := NewPTXAudioTower(model, rows)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range []int{13, rows, 13} {
		got, e := tower.ForwardRows(context.Background(), input[:n*projectedWidth], n)
		if e != nil {
			tower.Close()
			t.Fatal(e)
		}
		ref, e := model.ForwardOffline(input[:n*projectedWidth], n)
		if e != nil {
			tower.Close()
			t.Fatal(e)
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
		t.Logf("rows=%d max_abs=%g mean_abs=%g outside=%d", n, maxAbs, mean, outside)
		if outside != 0 || mean > 2e-5 {
			tower.Close()
			t.Fatalf("PTX max-row tower differs from CPU composition at %d", n)
		}
	}
	for i := range input {
		if input[i] != original[i] {
			tower.Close()
			t.Fatalf("input mutated at %d", i)
		}
	}
	tower.Close()
	after := ptx.StatsSnapshot()
	if after.Mallocs-before.Mallocs != after.Frees-before.Frees || after.MallocBytes-before.MallocBytes != after.FreeBytes-before.FreeBytes {
		t.Fatalf("PTX max-row tower leaked: before=%+v after=%+v", before, after)
	}
}
