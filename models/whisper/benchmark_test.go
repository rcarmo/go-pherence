package whisper

import (
	"math"
	"os"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/loader/audio"
)

// BenchmarkWhisperTinyRTF measures the real-time factor for whisper-tiny on CPU.
// RTF < 1.0 means faster than real-time.
func BenchmarkWhisperTinyRTF(b *testing.B) {
	modelPath := "../../models/whisper-tiny-hf/model.safetensors"
	if _, err := os.Stat(modelPath); err != nil {
		b.Skip("whisper-tiny model not available")
	}

	cfg := Tiny()
	enc, dec, err := LoadModel(modelPath, cfg)
	if err != nil {
		b.Fatalf("LoadModel: %v", err)
	}

	// Generate 3 seconds of speech-like audio
	const audioDuration = 3.0
	const sampleRate = 16000
	samples := make([]float32, int(audioDuration*sampleRate))
	for i := range samples {
		t := float64(i) / sampleRate
		samples[i] = float32(0.3*math.Sin(2*math.Pi*300*t) + 0.2*math.Sin(2*math.Pi*600*t))
	}

	// Precompute mel
	melCfg := audio.MelConfig{
		SampleRate: sampleRate,
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

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Encoder
		encoderOutput := enc.Forward(melFlat, T)
		encLen := len(encoderOutput) / cfg.EncoderDModel

		// Decoder (greedy, no timestamps)
		state := NewDecoderState(cfg, encoderOutput, encLen, dec)
		GreedyDecode(dec, state, cfg)
	}
	b.StopTimer()

	// Report RTF
	elapsed := b.Elapsed()
	rtf := elapsed.Seconds() / (float64(b.N) * audioDuration)
	b.ReportMetric(rtf, "RTF")
	b.ReportMetric(audioDuration/elapsed.Seconds()*float64(b.N), "x-realtime")
}

// BenchmarkWhisperTinyEncoder measures encoder-only throughput.
func BenchmarkWhisperTinyEncoder(b *testing.B) {
	modelPath := "../../models/whisper-tiny-hf/model.safetensors"
	if _, err := os.Stat(modelPath); err != nil {
		b.Skip("whisper-tiny model not available")
	}

	cfg := Tiny()
	enc, _, err := LoadModel(modelPath, cfg)
	if err != nil {
		b.Fatalf("LoadModel: %v", err)
	}

	// 3 seconds of mel input
	T := 187 // (3*16000 - 400) / 160 = ~187 frames
	melFlat := make([]float32, cfg.NumMelBins*T)
	for i := range melFlat {
		melFlat[i] = float32(i%80) * 0.01
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		enc.Forward(melFlat, T)
	}
}

// BenchmarkWhisperTinyDecoder measures decoder per-token throughput.
func BenchmarkWhisperTinyDecoder(b *testing.B) {
	modelPath := "../../models/whisper-tiny-hf/model.safetensors"
	if _, err := os.Stat(modelPath); err != nil {
		b.Skip("whisper-tiny model not available")
	}

	cfg := Tiny()
	enc, dec, err := LoadModel(modelPath, cfg)
	if err != nil {
		b.Fatalf("LoadModel: %v", err)
	}

	// Precompute encoder output
	T := 100
	melFlat := make([]float32, cfg.NumMelBins*T)
	encoderOutput := enc.Forward(melFlat, T)
	encLen := len(encoderOutput) / cfg.EncoderDModel

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		state := NewDecoderState(cfg, encoderOutput, encLen, dec)
		// Generate 10 tokens
		prevTok := TokenNoTimestamps
		for t := 0; t < 10; t++ {
			logits := dec.ForwardToken(prevTok, state)
			prevTok = argmax(logits)
		}
	}
}

// BenchmarkMelSpectrogram30s measures mel computation time for a full Whisper chunk.
func BenchmarkMelSpectrogram30s(b *testing.B) {
	samples := make([]float32, 30*16000)
	for i := range samples {
		samples[i] = float32(math.Sin(float64(i) * 0.01))
	}
	melCfg := audio.DefaultMelConfig()

	b.ResetTimer()
	var melResult [][]float32
	for i := 0; i < b.N; i++ {
		melResult = audio.MelSpectrogram(samples, melCfg)
	}
	_ = melResult
}

// TestRTFEstimate provides a quick RTF estimate without full benchmark iterations.
func TestRTFEstimate(t *testing.T) {
	modelPath := "../../models/whisper-tiny-hf/model.safetensors"
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("whisper-tiny model not available")
	}

	cfg := Tiny()
	enc, dec, err := LoadModel(modelPath, cfg)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	w := &Whisper{Encoder: enc, Decoder: dec, Config: cfg}

	// 2 seconds of audio
	const audioDuration = 2.0
	samples := make([]float32, int(audioDuration*16000))
	for i := range samples {
		samples[i] = float32(0.3 * math.Sin(2*math.Pi*440*float64(i)/16000))
	}

	start := time.Now()
	_, err = w.TranscribeFromSamples(samples)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("TranscribeFromSamples: %v", err)
	}

	rtf := elapsed.Seconds() / audioDuration
	t.Logf("CPU RTF: %.2f (%.1fs to transcribe %.1fs audio)", rtf, elapsed.Seconds(), audioDuration)
	t.Logf("Speed: %.2fx slower than real-time", rtf)
}
