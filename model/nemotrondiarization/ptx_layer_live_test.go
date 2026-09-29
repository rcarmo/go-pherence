package nemotrondiarization

import (
	"fmt"
	"math"
	"os"
	"sync"
	"testing"

	ptx "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedPTXAudioLayerParityAndLifetime(t *testing.T) {
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
	layer, err := NewPTXAudioLayer(layer1, 138)
	if err != nil {
		t.Fatal(err)
	}
	defer layer.Close()
	stacked := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*projectedWidth)
	for _, rows := range []int{16, 138, 16} {
		t.Run(fmt.Sprintf("rows=%d", rows), func(t *testing.T) {
			hidden, err := layer0.ForwardOffline(stacked[:rows*projectedWidth], rows)
			if err != nil {
				t.Fatal(err)
			}
			input := append([]float32(nil), hidden...)
			got, err := layer.Forward(hidden, rows)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(hidden) || &got[0] == &hidden[0] {
				t.Fatal("layer output aliases input")
			}
			for i := range hidden {
				if hidden[i] != input[i] {
					t.Fatalf("input mutated at %d", i)
				}
			}
			name := "testdata/jfk_layer1_complete.f32.gz"
			if rows == 138 {
				name = "testdata/jfk_full_layer1_complete.f32.gz"
			}
			ref := readStackingFixture(t, name, len(got))
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
				t.Fatal("PTX layer differs from PyTorch")
			}
		})
	}
	input := make([]float32, projectedWidth)
	for _, tc := range []struct {
		name string
		data []float32
		rows int
	}{
		{"zero", input, 0}, {"too_many", input, 139}, {"short", input[:1], 1},
		{"nan", func() []float32 { v := append([]float32(nil), input...); v[0] = float32(math.NaN()); return v }(), 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := layer.Forward(tc.data, tc.rows); err == nil {
				t.Fatal("accepted malformed input")
			}
		})
	}
	// A direct device-buffer call retains its input and weights on GPU.
	ptx.SetStatsEnabled(true)
	defer ptx.SetStatsEnabled(false)
	in, err := ptx.Malloc(len(input))
	if err != nil {
		t.Fatal(err)
	}
	defer in.Free()
	out, err := ptx.Malloc(len(input))
	if err != nil {
		t.Fatal(err)
	}
	defer out.Free()
	if err := in.Upload(input); err != nil {
		t.Fatal(err)
	}
	before := ptx.StatsSnapshot()
	if err := layer.ForwardBuffer(out, in, 1); err != nil {
		t.Fatal(err)
	}
	after := ptx.StatsSnapshot()
	if after.HostToDevice != before.HostToDevice || after.DeviceToHost != before.DeviceToHost {
		t.Fatal("device-resident layer transferred data during forward")
	}
	if after.Mallocs-before.Mallocs != after.Frees-before.Frees {
		t.Fatalf("device-resident layer leaked scratch: alloc=%d free=%d", after.Mallocs-before.Mallocs, after.Frees-before.Frees)
	}
	layer.Close()
	layer.Close()
	if _, err := layer.Forward(input, 1); err == nil {
		t.Fatal("accepted closed layer")
	}
}

func TestPTXAudioLayerRejectMalformedWithoutCUDA(t *testing.T) {
	if _, err := NewPTXAudioLayer(nil, 1); err == nil {
		t.Fatal("accepted nil model")
	}
	if err := (*PTXAudioLayer)(nil).ForwardBuffer(nil, nil, 1); err == nil {
		t.Fatal("accepted nil receiver")
	}
	layer := &PTXAudioLayer{maxRows: 2}
	for _, tc := range []struct {
		name    string
		out, in *ptx.Buffer
		rows    int
	}{
		{"nil", nil, nil, 1},
		{"zero", &ptx.Buffer{}, &ptx.Buffer{}, 1},
		{"zero_rows", &ptx.Buffer{}, &ptx.Buffer{}, 0},
		{"short", &ptx.Buffer{Ptr: 0x10000, Size: 4}, &ptx.Buffer{Ptr: 0x20000, Size: 2048}, 1},
		{"same_buffer", &ptx.Buffer{Ptr: 0x10000, Size: 2048}, &ptx.Buffer{Ptr: 0x10000, Size: 2048}, 1},
		{"overlap", &ptx.Buffer{Ptr: 0x10004, Size: 2048}, &ptx.Buffer{Ptr: 0x10000, Size: 2048}, 1},
		{"too_many", &ptx.Buffer{Ptr: 0x10000, Size: 4096}, &ptx.Buffer{Ptr: 0x20000, Size: 4096}, 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := layer.ForwardBuffer(tc.out, tc.in, tc.rows); err == nil {
				t.Fatal("accepted malformed device input")
			}
		})
	}
}

func TestReleasedPTXAudioLayerSharedConcurrency(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	}
	if !ptx.SgemmReady() {
		t.Skip("CUDA unavailable")
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
	layer, err := NewPTXAudioLayer(model, 16)
	if err != nil {
		t.Fatal(err)
	}
	defer layer.Close()
	input := make([]float32, 16*projectedWidth)
	for i := range input {
		input[i] = float32((i%29)-14) / 8
	}
	want, err := layer.Forward(input, 16)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	errs := make(chan error, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, err := layer.Forward(input, 16)
			if err != nil {
				errs <- err
				return
			}
			for j := range got {
				if got[j] != want[j] {
					errs <- fmt.Errorf("shared PTX layer drift at %d", j)
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
}
