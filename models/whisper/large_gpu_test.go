package whisper

import (
	"os"
	"testing"
	"time"

	nv "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/audio"
)

// TestLargeV3GPUEncoder loads whisper-large-v3 and benchmarks the GPU encoder on real audio.
func TestLargeV3GPUEncoder(t *testing.T) {
	modelPath := "../../models/whisper-large-v3-hf/model.safetensors"
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("whisper-large-v3 model not available")
	}
	audioPath := "../../testdata/jfk.wav"
	if _, err := os.Stat(audioPath); err != nil {
		t.Skip("jfk.wav test audio not available")
	}
	if !nv.SgemmReady() {
		t.Skip("GPU not available")
	}

	cfg := LargeV3()
	t.Logf("Loading whisper-large-v3 (%d enc layers, %d dec layers, d=%d)...", cfg.EncoderLayers, cfg.DecoderLayers, cfg.EncoderDModel)

	start := time.Now()
	enc, _, err := LoadModel(modelPath, cfg)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	t.Logf("Model loaded in %v", time.Since(start))

	// Load real audio
	samples, sampleRate, err := audio.WAV(audioPath)
	if err != nil {
		t.Fatalf("WAV: %v", err)
	}
	if sampleRate != 16000 {
		samples = audio.ResampleSinc(samples, sampleRate, 16000)
	}
	audioDuration := float64(len(samples)) / 16000
	t.Logf("Audio: %.1fs, %d samples", audioDuration, len(samples))

	// Compute mel spectrogram
	melCfg := audio.MelConfig{
		SampleRate: 16000,
		FFTSize:    400,
		HopLength:  160,
		NumMels:    cfg.NumMelBins,
		NFFTPadded: 512,
	}
	mel := audio.MelSpectrogram(samples, melCfg)
	if mel == nil {
		t.Fatal("MelSpectrogram returned nil")
	}
	T := len(mel[0])
	melFlat := make([]float32, cfg.NumMelBins*T)
	for m := 0; m < cfg.NumMelBins; m++ {
		copy(melFlat[m*T:], mel[m])
	}
	t.Logf("Mel: %d bins × %d frames", cfg.NumMelBins, T)

	// GPU encoder
	gpuEnc := NewGPUEncoder(enc, cfg)
	if !gpuEnc.ready {
		t.Skip("GPU encoder not ready")
	}
	t.Logf("GPU encoder initialized, weights uploaded")

	// Benchmark: GPU encoder forward
	start = time.Now()
	out := gpuEnc.ForwardGPU(melFlat, T)
	gpuTime := time.Since(start)

	encT := len(out) / cfg.EncoderDModel
	t.Logf("GPU Encoder: %v for %.1fs audio (T'=%d)", gpuTime, audioDuration, encT)
	t.Logf("GPU Encoder RTF: %.3f", gpuTime.Seconds()/audioDuration)

	// CPU encoder for comparison
	start = time.Now()
	cpuOut := enc.Forward(melFlat, T)
	cpuTime := time.Since(start)
	t.Logf("CPU Encoder: %v (RTF: %.3f)", cpuTime, cpuTime.Seconds()/audioDuration)
	t.Logf("GPU speedup: %.2fx", float64(cpuTime)/float64(gpuTime))

	// Verify outputs match
	if len(out) != len(cpuOut) {
		t.Fatalf("length mismatch: gpu=%d cpu=%d", len(out), len(cpuOut))
	}
	var maxDiff float32
	for i := range out {
		d := out[i] - cpuOut[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	t.Logf("Max GPU vs CPU diff: %e", maxDiff)
}
