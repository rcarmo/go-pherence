package nemotrondiarization

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"sync"
	"testing"

	ptx "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedPTXAudioTowerPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL to pinned model.safetensors")
	}
	if !ptx.SgemmReady() {
		t.Skip("CUDA unavailable")
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
	tower, err := NewPTXAudioTower(model, 138)
	if err != nil {
		t.Fatal(err)
	}
	defer tower.Close()
	stacked := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*projectedWidth)
	for _, rows := range []int{16, 138, 16} {
		t.Run(fmt.Sprintf("rows=%d", rows), func(t *testing.T) {
			input := append([]float32(nil), stacked[:rows*projectedWidth]...)
			original := append([]float32(nil), input...)
			ptx.SetStatsEnabled(true)
			defer ptx.SetStatsEnabled(false)
			beforeStats := ptx.StatsSnapshot()
			out, err := tower.ForwardRows(context.Background(), input, rows)
			afterStats := ptx.StatsSnapshot()
			if err != nil {
				t.Fatal(err)
			}
			if afterStats.HostToDevice-beforeStats.HostToDevice != 1 || afterStats.DeviceToHost-beforeStats.DeviceToHost != 1 ||
				afterStats.Mallocs-beforeStats.Mallocs != afterStats.Frees-beforeStats.Frees {
				t.Fatalf("unexpected PTX tower transfers or scratch lifetime: before=%+v after=%+v", beforeStats, afterStats)
			}
			if len(out) != len(input) || &out[0] == &input[0] {
				t.Fatal("output ownership")
			}
			for i := range input {
				if input[i] != original[i] {
					t.Fatalf("input mutated at %d", i)
				}
			}
			if rows == 138 {
				compareTowerFixture(t, "tower_normal", out)
			} else {
				// No independent 16-row final norm is pinned yet. Compare to
				// the separately composed CPU tower without calling this quality evidence.
				ref, err := model.ForwardOffline(input, rows)
				if err != nil {
					t.Fatal(err)
				}
				var max float64
				for i, v := range out {
					max = math.Max(max, math.Abs(float64(v-ref[i])))
				}
				t.Logf("16-row CPU composition max_abs=%g", max)
				if max > 3e-4 {
					t.Fatal("16-row PTX tower differs from CPU composition")
				}
			}
		})
	}
	// Exact 13-row attention keys must remain distinct from the 138-row
	// capacity after mixed-size calls. This is CPU-composition parity only.
	shortInput := stacked[:13*projectedWidth]
	short, err := tower.ForwardRows(context.Background(), shortInput, 13)
	if err != nil {
		t.Fatal(err)
	}
	shortRef, err := model.ForwardOffline(shortInput, 13)
	if err != nil {
		t.Fatal(err)
	}
	var shortMax float64
	for i, v := range short {
		shortMax = math.Max(shortMax, math.Abs(float64(v-shortRef[i])))
	}
	t.Logf("13-row CPU composition max_abs=%g", shortMax)
	if shortMax > 3e-4 {
		t.Fatal("short PTX prefix differs from CPU composition")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if out, err := tower.ForwardRows(ctx, stacked[:16*projectedWidth], 16); !errors.Is(err, context.Canceled) || out != nil {
		t.Fatalf("cancelled request out=%d err=%v", len(out), err)
	}
	for _, tc := range []struct {
		rows  int
		input []float32
	}{{0, nil}, {139, stacked}, {16, stacked[:1]}} {
		if out, err := tower.ForwardRows(context.Background(), tc.input, tc.rows); err == nil || out != nil {
			t.Fatal("accepted invalid tower input")
		}
	}
	// Two callers share one tower. Serial execution avoids unsafe weight
	// teardown and keeps each request's output separately owned.
	ref, err := tower.ForwardRows(context.Background(), stacked[:16*projectedWidth], 16)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	errs := make(chan error, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, err := tower.ForwardRows(context.Background(), stacked[:16*projectedWidth], 16)
			if err != nil {
				errs <- err
				return
			}
			for j, value := range got {
				if value != ref[j] {
					errs <- fmt.Errorf("shared PTX tower drift at %d", j)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	tower.Close()
	if out, err := tower.ForwardRows(context.Background(), stacked[:16*projectedWidth], 16); err == nil || out != nil {
		t.Fatal("accepted closed tower")
	}
}

func TestPTXAudioTowerRejectsMalformed(t *testing.T) {
	if _, err := NewPTXAudioTower(nil, 1); err == nil {
		t.Fatal("accepted nil model")
	}
	if _, err := (*PTXAudioTower)(nil).ForwardRows(context.Background(), make([]float32, projectedWidth), 1); err == nil {
		t.Fatal("accepted nil tower")
	}
}
