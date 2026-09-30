package nemotrondiarization

import (
	"os"
	"testing"
	"time"

	ptx "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// One exact 103-row prepared window, with each stage measured separately.
// This is diagnostic CPU/GPU wall time, not a complete PCM request benchmark.
func TestPTX103StageDiagnostic(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_NEMOTRON_PTX_STAGES") != "1" {
		t.Skip("set GO_PHERENCE_TEST_NEMOTRON_PTX_STAGES=1")
	}
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set model path")
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
	head, err := LoadOfflineHead(file)
	if err != nil {
		t.Fatal(err)
	}
	if err = file.Close(); err != nil {
		t.Fatal(err)
	}
	tower, err := NewPTXAudioTower(model, 103)
	if err != nil {
		t.Fatal(err)
	}
	defer tower.Close()
	stacked := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*projectedWidth)
	input := stacked[:103*projectedWidth]
	for trial := 0; trial < 3; trial++ {
		started := time.Now()
		first, err := model.first.ForwardOffline(input, 103)
		if err != nil {
			t.Fatal(err)
		}
		firstTime := time.Since(started)
		in, err := ptx.Malloc(len(first))
		if err != nil {
			t.Fatal(err)
		}
		out, err := ptx.Malloc(len(first))
		if err != nil {
			in.Free()
			t.Fatal(err)
		}
		started = time.Now()
		if err = in.Upload(first); err != nil {
			t.Fatal(err)
		}
		uploadTime := time.Since(started)
		started = time.Now()
		a, b := in, out
		for i, l := range tower.layers {
			if err = l.forwardWithScratch(b, a, 103, tower.scratch); err != nil {
				t.Fatalf("layer %d: %v", i, err)
			}
			a, b = b, a
		}
		if err = ptx.AffineLayerNormF32Buffer(b, a, tower.weight, tower.bias, 103, projectedWidth, 1e-5); err != nil {
			t.Fatal(err)
		}
		if err = ptx.SyncErr(); err != nil {
			t.Fatal(err)
		}
		gpuTime := time.Since(started)
		hidden := make([]float32, len(first))
		started = time.Now()
		if err = b.Download(hidden); err != nil {
			t.Fatal(err)
		}
		downloadTime := time.Since(started)
		started = time.Now()
		logits, err := head.ForwardOffline(hidden, 103)
		if err != nil {
			t.Fatal(err)
		}
		headTime := time.Since(started)
		if len(logits) != 103*diarizationUpsample*diarizationSpeakers {
			t.Fatal("wrong head shape")
		}
		if err = ptx.SyncErr(); err != nil {
			t.Fatal(err)
		}
		in.Free()
		out.Free()
		t.Logf("trial=%d cpu_layer0=%s h2d=%s gpu_30_layers_and_final=%s d2h=%s cpu_head=%s", trial, firstTime, uploadTime, gpuTime, downloadTime, headTime)
	}
}
