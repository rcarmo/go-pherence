package nemotrondiarization

import (
	"fmt"
	"math"
	"os"
	"testing"

	ptx "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// The norm fixture is produced by the released CPU PyTorch model, not by the
// Go LayerNorm. Only the layer-0 hidden state is composed in Go here.
func TestReleasedLayer1AffinePTXLayerNormParity(t *testing.T) {
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
	gamma, _, err := file.GetFloat32("model.audio_tower.layers.1.layer_norm1.weight")
	if err != nil {
		t.Fatal(err)
	}
	beta, _, err := file.GetFloat32("model.audio_tower.layers.1.layer_norm1.bias")
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if len(gamma) != projectedWidth || len(beta) != projectedWidth {
		t.Fatal("unexpected layer-1 affine weights")
	}
	stacked := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*projectedWidth)
	for _, rows := range []int{16, 138} {
		t.Run(fmt.Sprintf("rows=%d", rows), func(t *testing.T) {
			hidden, err := layer0.ForwardOffline(stacked[:rows*projectedWidth], rows)
			if err != nil {
				t.Fatal(err)
			}
			input := append([]float32(nil), hidden...)
			hiddenFixture := "testdata/jfk_layer0_complete.f32.gz"
			if rows == 138 {
				hiddenFixture = "testdata/jfk_full_layer0_complete.f32.gz"
			}
			hiddenRef := readStackingFixture(t, hiddenFixture, len(hidden))
			for i, value := range hidden {
				if math.Abs(float64(value-hiddenRef[i])) > 3e-4+2e-5*math.Abs(float64(hiddenRef[i])) {
					t.Fatalf("layer-0 input differs from PyTorch at %d", i)
				}
			}
			name := "testdata/jfk_layer1_normal.f32.gz"
			if rows == 138 {
				name = "testdata/jfk_full_layer1_normal.f32.gz"
			}
			ref := readStackingFixture(t, name, len(hidden))
			out := make([]float32, len(hidden))
			bOut, bX, bGamma, bBeta := ptx.NewDevBufFrom(out), ptx.NewDevBufFrom(hidden), ptx.NewDevBufFrom(gamma), ptx.NewDevBufFrom(beta)
			defer bOut.Free()
			defer bX.Free()
			defer bGamma.Free()
			defer bBeta.Free()
			for _, b := range []*ptx.DevBuf{bOut, bX, bGamma, bBeta} {
				if err := b.EnsureGPU(); err != nil {
					t.Fatal(err)
				}
			}
			if err := ptx.AffineLayerNormF32Buffer(bOut.GPUBuffer(), bX.GPUBuffer(), bGamma.GPUBuffer(), bBeta.GPUBuffer(), rows, projectedWidth, 1e-5); err != nil {
				t.Fatal(err)
			}
			if err := ptx.SyncErr(); err != nil {
				t.Fatal(err)
			}
			if err := bOut.GPUBuffer().Download(out); err != nil {
				t.Fatal(err)
			}
			deviceInput := make([]float32, len(hidden))
			if err := bX.GPUBuffer().Download(deviceInput); err != nil {
				t.Fatal(err)
			}
			for i := range input {
				if input[i] != hidden[i] || input[i] != deviceInput[i] {
					t.Fatalf("input mutated at %d", i)
				}
			}
			var maxAbs, sumAbs float64
			var outside int
			for i, value := range out {
				delta := math.Abs(float64(value - ref[i]))
				maxAbs = math.Max(maxAbs, delta)
				sumAbs += delta
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(ref[i])) {
					outside++
				}
			}
			mean := sumAbs / float64(len(out))
			t.Logf("max_abs=%g mean_abs=%g outside=%d", maxAbs, mean, outside)
			if outside != 0 || mean > 2e-5 {
				t.Fatal("PTX layer-1 norm differs from PyTorch")
			}
		})
	}
}
