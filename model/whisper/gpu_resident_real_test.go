package whisper

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/weights"
)

func TestMOSSRealWeightsResidentLayerDrift(t *testing.T) {
	modelDir := os.Getenv("MOSS_TRANSCRIBE_MODEL_DIR")
	if modelDir == "" || os.Getenv("MOSS_TRANSCRIBE_GPU_PARITY") == "" {
		t.Skip("set MOSS_TRANSCRIBE_MODEL_DIR and MOSS_TRANSCRIBE_GPU_PARITY=1")
	}
	if _, err := os.Stat(filepath.Join(modelDir, "config.json")); err != nil {
		t.Skip(err)
	}
	source, err := weights.OpenSafetensors(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	cfg := Medium()
	enc, err := LoadEncoderSource(source, "model.whisper_encoder", cfg)
	if err != nil {
		t.Fatal(err)
	}
	gpu := NewGPUEncoder(enc, cfg)
	defer gpu.Close()
	if !gpu.Ready() {
		t.Skip("runtime-loaded NVIDIA PTX unavailable")
	}
	t.Setenv("GO_PHERENCE_WHISPER_GPU_RESIDENT", "1")
	t.Setenv("GO_PHERENCE_WHISPER_GPU_ATTENTION", "0")

	const seqLen = 64
	xCPU := make([]float32, seqLen*cfg.EncoderDModel)
	for i := range xCPU {
		xCPU[i] = float32(0.1*math.Sin(float64(i)*0.013) + 0.03*math.Cos(float64(i)*0.031))
	}
	xGPU := append([]float32(nil), xCPU...)
	for layerIndex := range enc.Layers {
		want := enc.forwardLayer(layerIndex, &enc.Layers[layerIndex], xCPU, seqLen)
		got, ok := gpu.forwardLayerGPUResident(layerIndex, xGPU, seqLen)
		if !ok {
			t.Fatalf("layer %d resident path fell back", layerIndex)
		}
		maxDiff, maxIndex := maxAbsDiff(got, want)
		t.Logf("layer=%d max_abs_diff=%.6g index=%d", layerIndex, maxDiff, maxIndex)
		if maxDiff > 0.03 {
			t.Fatalf("layer %d max abs diff %.6g exceeds 0.03", layerIndex, maxDiff)
		}
		xCPU, xGPU = want, got
	}
	maxDiff, maxIndex := maxAbsDiff(xGPU, xCPU)
	t.Logf("sequential max_abs_diff=%.6g index=%d", maxDiff, maxIndex)
}

func maxAbsDiff(got, want []float32) (float64, int) {
	maxDiff, maxIndex := float64(0), -1
	for i := range got {
		if diff := math.Abs(float64(got[i] - want[i])); diff > maxDiff {
			maxDiff, maxIndex = diff, i
		}
	}
	return maxDiff, maxIndex
}
