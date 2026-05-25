package whisper

import (
	"math"
	"os"
	"testing"
)

func TestTokenizerLoad(t *testing.T) {
	tokPath := "../../models/whisper-tiny-hf/tokenizer.json"
	if _, err := os.Stat(tokPath); err != nil {
		t.Skip("tokenizer.json not available")
	}

	tok, err := LoadTokenizer(tokPath)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	t.Logf("Loaded tokenizer: %d tokens", tok.VocabSize)

	// Test decoding known tokens
	// Token 264 = "Ġthe" → " the"
	text := tok.Decode([]int{264})
	if text != "the" { // trimmed
		t.Logf("token 264 decoded to: %q", text)
	}

	// Token 0 = "!"
	text = tok.Decode([]int{0})
	if text != "!" {
		t.Fatalf("token 0 = %q want '!'", text)
	}
}

func TestTokenizerWithModel(t *testing.T) {
	modelPath := "../../models/whisper-tiny-hf/model.safetensors"
	tokPath := "../../models/whisper-tiny-hf/tokenizer.json"
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("model not available")
	}
	if _, err := os.Stat(tokPath); err != nil {
		t.Skip("tokenizer not available")
	}

	// Load tokenizer globally
	if err := LoadTokenizerGlobal(tokPath); err != nil {
		t.Fatalf("LoadTokenizerGlobal: %v", err)
	}
	defer func() { DefaultTokenizer = nil }()

	cfg := Tiny()
	enc, dec, err := LoadModel(modelPath, cfg)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	w := &Whisper{Encoder: enc, Decoder: dec, Config: cfg}

	// 2 seconds of speech-like audio
	samples := make([]float32, 2*16000)
	for i := range samples {
		tt := float64(i) / 16000
		samples[i] = float32(0.3*math.Sin(2*math.Pi*300*tt) + 0.2*math.Sin(2*math.Pi*600*tt))
	}

	text, err := w.TranscribeFromSamples(samples)
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	t.Logf("Transcription with tokenizer: %q", text)

	// Should produce actual text, not [tok] placeholders
	if len(text) > 0 && text[0] == '[' {
		t.Fatalf("still getting placeholder output: %q", text[:min(50, len(text))])
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
