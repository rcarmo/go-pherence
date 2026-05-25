package whisper

import (
	"math"
	"os"
	"testing"
	"time"

	nv "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/audio"
)

func TestGPURTFEstimate(t *testing.T) {
	modelPath := "../../models/whisper-tiny-hf/model.safetensors"
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("whisper-tiny model not available")
	}
	if !nv.SgemmReady() {
		t.Skip("GPU not available")
	}

	cfg := Tiny()
	enc, dec, err := LoadModel(modelPath, cfg)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	gpuEnc := NewGPUEncoder(enc, cfg)

	// 2 seconds of speech-like audio
	const audioDuration = 2.0
	samples := make([]float32, int(audioDuration*16000))
	for i := range samples {
		tt := float64(i) / 16000
		samples[i] = float32(0.3*math.Sin(2*math.Pi*300*tt) + 0.2*math.Sin(2*math.Pi*600*tt))
	}

	// Compute mel
	melCfg := audio.MelConfig{
		SampleRate: 16000,
		FFTSize:    400,
		HopLength:  160,
		NumMels:    cfg.NumMelBins,
		NFFTPadded: 512,
	}
	mel := audio.MelSpectrogram(samples, melCfg)
	T := len(mel[0])
	melFlat := make([]float32, cfg.NumMelBins*T)
	for m := 0; m < cfg.NumMelBins; m++ {
		copy(melFlat[m*T:], mel[m])
	}

	start := time.Now()

	// GPU encoder
	encoderOutput := gpuEnc.ForwardGPU(melFlat, T)
	encLen := len(encoderOutput) / cfg.EncoderDModel

	// Decoder (CPU — would need GPU DevBuf integration for full GPU path)
	state := NewDecoderStateGPU(cfg, encoderOutput, encLen, dec)
	GreedyDecode(dec, state, cfg)

	elapsed := time.Since(start)
	rtf := elapsed.Seconds() / audioDuration

	t.Logf("GPU Encoder + CPU Decoder RTF: %.2f (%.1fs for %.1fs audio)", rtf, elapsed.Seconds(), audioDuration)
	t.Logf("Speedup vs pure CPU (RTF=6.3): %.2fx", 6.3/rtf)
}
