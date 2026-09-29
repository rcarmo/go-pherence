package nemotrondiarization

import (
	"fmt"
	"math"
	"os"
	"testing"

	ptx "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// The attention residual is composed on CPU from the released model. All MLP
// activations, including both affine norms, remain on GPU until final download.
// This isolates the resident MLP path; it does not qualify a resident layer.
func TestReleasedLayer1PTXResidentMLPParity(t *testing.T) {
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
	// Sgemm consumes row-major [K,N] weights. The released checkpoint stores
	// PyTorch [N,K] matrices; transpose once before any GPU request.
	transpose := func(weight []float32, n, k int) []float32 {
		out := make([]float32, len(weight))
		for j := 0; j < n; j++ {
			for i := 0; i < k; i++ {
				out[i*n+j] = weight[j*k+i]
			}
		}
		return out
	}
	stacked := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*projectedWidth)
	for _, rows := range []int{16, 138} {
		t.Run(fmt.Sprintf("rows=%d", rows), func(t *testing.T) {
			hidden, err := layer0.ForwardOffline(stacked[:rows*projectedWidth], rows)
			if err != nil {
				t.Fatal(err)
			}
			residual, err := layer1.attention.forwardResidual(hidden, rows)
			if err != nil {
				t.Fatal(err)
			}
			original := append([]float32(nil), residual...)
			upload := func(data []float32) *ptx.Buffer {
				b, err := ptx.Malloc(len(data))
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(b.Free)
				if err := b.Upload(data); err != nil {
					t.Fatal(err)
				}
				return b
			}
			alloc := func(n int) *ptx.Buffer {
				b, err := ptx.Malloc(n)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(b.Free)
				return b
			}
			dResidual := upload(residual)
			dNorm, dFC1, dFC2 := alloc(len(residual)), alloc(rows*diarizationIntermediate), alloc(len(residual))
			dNW, dNB := upload(layer1.normWeight), upload(layer1.normBias)
			dW1, dB1 := upload(transpose(layer1.fc1Weight, diarizationIntermediate, projectedWidth)), upload(layer1.fc1Bias)
			dW2, dB2 := upload(transpose(layer1.fc2Weight, projectedWidth, diarizationIntermediate)), upload(layer1.fc2Bias)
			if err := ptx.AffineLayerNormF32Buffer(dNorm, dResidual, dNW, dNB, rows, projectedWidth, 1e-5); err != nil {
				t.Fatal(err)
			}
			if err := ptx.Sgemm(rows, diarizationIntermediate, projectedWidth, 1, dNorm, dW1, dFC1); err != nil {
				t.Fatal(err)
			}
			if err := ptx.WhisperRowBiasBuffer(dFC1, dB1, rows, diarizationIntermediate); err != nil {
				t.Fatal(err)
			}
			if err := ptx.GELUErfF32Buffer(dFC1, rows*diarizationIntermediate); err != nil {
				t.Fatal(err)
			}
			if err := ptx.Sgemm(rows, projectedWidth, diarizationIntermediate, 1, dFC1, dW2, dFC2); err != nil {
				t.Fatal(err)
			}
			if err := ptx.WhisperRowBiasBuffer(dFC2, dB2, rows, projectedWidth); err != nil {
				t.Fatal(err)
			}
			if err := ptx.VecAddF32Buffer(dResidual, dFC2, dFC2, len(residual)); err != nil {
				t.Fatal(err)
			}
			if err := ptx.SyncErr(); err != nil {
				t.Fatal(err)
			}
			got := make([]float32, len(residual))
			if err := dFC2.Download(got); err != nil {
				t.Fatal(err)
			}
			readOnly := make([]float32, len(residual))
			if err := dResidual.Download(readOnly); err != nil {
				t.Fatal(err)
			}
			for i := range residual {
				if residual[i] != original[i] || readOnly[i] != original[i] {
					t.Fatalf("residual mutated at %d", i)
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
				t.Fatal("resident PTX MLP differs from PyTorch layer-1 fixture")
			}
		})
	}
}
