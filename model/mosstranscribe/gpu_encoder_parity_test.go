package mosstranscribe

import (
	"math"
	"os"
	"testing"
)

func TestRealCheckpointWhisperGPUEncoderParity(t *testing.T) {
	modelDir := os.Getenv("MOSS_TRANSCRIBE_MODEL_DIR")
	if modelDir == "" {
		t.Skip("set MOSS_TRANSCRIBE_MODEL_DIR for the pinned GPU parity gate")
	}
	if os.Getenv("MOSS_TRANSCRIBE_GPU_PARITY") == "" {
		t.Skip("set MOSS_TRANSCRIBE_GPU_PARITY=1 to run the CPU/GPU encoder comparison")
	}
	model, err := LoadAudioBackbone(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	if !model.EnableGPU() {
		t.Skip("runtime-loaded NVIDIA PTX backend unavailable")
	}

	samples := make([]float32, AudioChunkSamples)
	for i := 0; i < AudioSampleRate; i++ {
		samples[i] = float32(0.2*math.Sin(2*math.Pi*440*float64(i)/AudioSampleRate) +
			0.05*math.Sin(2*math.Pi*(200+1200*float64(i)/AudioSampleRate)*float64(i)/AudioSampleRate))
	}
	cfg := model.Config.WhisperConfig()
	features, err := (AudioChunk{Samples: samples, TokenLength: 375}).InputFeatures(cfg)
	if err != nil {
		t.Fatal(err)
	}
	want := model.Encoder.Forward(features, cfg.MaxLength)
	t.Setenv("GO_PHERENCE_WHISPER_GPU_RESIDENT", "1")
	got := model.GPUEncoder.ForwardGPU(features, cfg.MaxLength)
	if len(got) != len(want) {
		t.Fatalf("GPU encoder length=%d want %d", len(got), len(want))
	}
	var maxDiff float64
	maxIndex := -1
	for i := range want {
		diff := math.Abs(float64(got[i] - want[i]))
		if diff > maxDiff {
			maxDiff, maxIndex = diff, i
		}
	}
	// The resident kernel uses CUDA's fast tanh GELU approximation while the
	// CPU oracle uses erf GELU. This bound is tightened before default enablement.
	if maxDiff > 3e-3 {
		t.Fatalf("GPU encoder max abs diff %.6g at %d exceeds 3e-3", maxDiff, maxIndex)
	}
	t.Logf("GPU encoder max abs diff %.6g at %d", maxDiff, maxIndex)
}
