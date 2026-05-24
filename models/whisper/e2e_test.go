package whisper

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio"
)

// TestWhisperTinyTranscribe runs the full pipeline with real weights on synthetic audio.
func TestWhisperTinyTranscribe(t *testing.T) {
	modelPath := "../../models/whisper-tiny-hf/model.safetensors"
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("whisper-tiny model not available:", err)
	}

	cfg := Tiny()
	enc, dec, err := LoadModel(modelPath, cfg)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	w := &Whisper{Encoder: enc, Decoder: dec, Config: cfg}

	// Generate 2 seconds of 440Hz tone (will produce silence/noise transcription)
	samples := make([]float32, 2*16000)
	for i := range samples {
		samples[i] = float32(0.3 * math.Sin(2*math.Pi*440*float64(i)/16000))
	}

	text, err := w.TranscribeFromSamples(samples)
	if err != nil {
		t.Fatalf("TranscribeFromSamples: %v", err)
	}
	t.Logf("transcription: %q", text)
	// With a pure tone, the model should produce something (even if it's silence/noise tokens)
}

// TestWhisperTinyMelPipeline verifies mel spectrogram → encoder works with real audio.
func TestWhisperTinyMelPipeline(t *testing.T) {
	modelPath := "../../models/whisper-tiny-hf/model.safetensors"
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("whisper-tiny model not available:", err)
	}

	cfg := Tiny()
	enc, _, err := LoadModel(modelPath, cfg)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	// Generate 3 seconds of speech-like audio (mixed frequencies)
	sampleRate := 16000
	duration := 3.0
	numSamples := int(duration * float64(sampleRate))
	samples := make([]float32, numSamples)
	for i := range samples {
		t := float64(i) / float64(sampleRate)
		// Simulate vowel-like spectrum
		samples[i] = float32(0.3*math.Sin(2*math.Pi*300*t) +
			0.2*math.Sin(2*math.Pi*600*t) +
			0.1*math.Sin(2*math.Pi*1200*t))
	}

	// Compute mel spectrogram
	melCfg := audio.DefaultMelConfig()
	melCfg.NumMels = cfg.NumMelBins
	mel := audio.MelSpectrogram(samples, melCfg)
	if mel == nil {
		t.Fatal("MelSpectrogram returned nil")
	}

	T := len(mel[0])
	melFlat := make([]float32, cfg.NumMelBins*T)
	for m := 0; m < cfg.NumMelBins; m++ {
		copy(melFlat[m*T:], mel[m])
	}

	// Run encoder
	out := enc.Forward(melFlat, T)
	expectedT := (T+2-3)/2 + 1
	if len(out) != expectedT*cfg.EncoderDModel {
		t.Fatalf("encoder output length=%d want %d", len(out), expectedT*cfg.EncoderDModel)
	}

	// Verify non-trivial output
	var maxAbs float32
	for _, v := range out {
		a := v
		if a < 0 {
			a = -a
		}
		if a > maxAbs {
			maxAbs = a
		}
	}
	if maxAbs < 0.1 {
		t.Fatalf("encoder output max_abs=%f, too small", maxAbs)
	}
	t.Logf("mel frames=%d, encoder output T'=%d, max_abs=%.3f", T, expectedT, maxAbs)
}

// TestWAVRoundtrip generates a WAV in memory and verifies the pipeline reads it correctly.
func TestWAVRoundtrip(t *testing.T) {
	const sampleRate = 16000
	const numSamples = 16000 // 1 second
	var buf bytes.Buffer

	// Write WAV header
	buf.WriteString("RIFF")
	binary.Write(&buf, binary.LittleEndian, uint32(36+numSamples*2))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	binary.Write(&buf, binary.LittleEndian, uint32(16))
	binary.Write(&buf, binary.LittleEndian, uint16(1))
	binary.Write(&buf, binary.LittleEndian, uint16(1))
	binary.Write(&buf, binary.LittleEndian, uint32(sampleRate))
	binary.Write(&buf, binary.LittleEndian, uint32(sampleRate*2))
	binary.Write(&buf, binary.LittleEndian, uint16(2))
	binary.Write(&buf, binary.LittleEndian, uint16(16))
	buf.WriteString("data")
	binary.Write(&buf, binary.LittleEndian, uint32(numSamples*2))
	for i := 0; i < numSamples; i++ {
		sample := int16(16000 * math.Sin(2*math.Pi*1000*float64(i)/sampleRate))
		binary.Write(&buf, binary.LittleEndian, sample)
	}

	samples, rate, err := audio.ReadWAV(&buf)
	if err != nil {
		t.Fatalf("ReadWAV: %v", err)
	}
	if rate != sampleRate {
		t.Fatalf("rate=%d want %d", rate, sampleRate)
	}
	if len(samples) != numSamples {
		t.Fatalf("samples=%d want %d", len(samples), numSamples)
	}

	// Verify mel works
	melCfg := audio.DefaultMelConfig()
	mel := audio.MelSpectrogram(samples, melCfg)
	if mel == nil || len(mel) != 80 {
		t.Fatal("mel spectrogram failed")
	}
	t.Logf("WAV roundtrip: %d samples → %d mel frames", len(samples), len(mel[0]))
}
