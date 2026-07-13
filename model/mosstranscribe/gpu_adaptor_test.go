package mosstranscribe

import (
	"math"
	"os"
	"testing"
)

func TestRealCheckpointGPUAdaptorParity(t *testing.T) {
	modelDir := os.Getenv("MOSS_TRANSCRIBE_MODEL_DIR")
	if modelDir == "" || os.Getenv("MOSS_TRANSCRIBE_GPU_PARITY") == "" {
		t.Skip("set MOSS_TRANSCRIBE_MODEL_DIR and MOSS_TRANSCRIBE_GPU_PARITY=1")
	}
	model, err := LoadAudioBackbone(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	gpu := NewGPUAdaptor(model.Adaptor)
	defer gpu.Close()
	if !gpu.Ready() {
		t.Skip("runtime-loaded NVIDIA PTX unavailable")
	}
	const tokens = 13
	merged := make([]float32, tokens*AdaptorInputDim)
	for i := range merged {
		merged[i] = float32(0.08*math.Sin(float64(i)*0.017) + 0.02*math.Cos(float64(i)*0.043))
	}
	want, scratch := make([]float32, tokens*AdaptorHiddenDim), make([]float32, tokens*AdaptorHiddenDim)
	if !ForwardAdaptorTo(want, scratch, merged, tokens, model.Adaptor) {
		t.Fatal("CPU adaptor failed")
	}
	got := make([]float32, len(want))
	if !gpu.Forward(got, merged, tokens) {
		t.Fatal("GPU adaptor failed")
	}
	maxDiff, maxIndex := maxAbsDiffMOSS(got, want)
	if maxDiff > 3e-4 {
		t.Fatalf("GPU adaptor max abs diff %.6g at %d exceeds 3e-4", maxDiff, maxIndex)
	}
	t.Logf("GPU adaptor max abs diff %.6g at %d", maxDiff, maxIndex)
}

func maxAbsDiffMOSS(got, want []float32) (float64, int) {
	maxDiff, maxIndex := float64(0), -1
	for i := range got {
		if diff := math.Abs(float64(got[i] - want[i])); diff > maxDiff {
			maxDiff, maxIndex = diff, i
		}
	}
	return maxDiff, maxIndex
}
