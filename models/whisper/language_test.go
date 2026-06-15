package whisper

import (
	"os"
	"testing"
)

func TestDetectLanguage(t *testing.T) {
	modelPath := "../../models/whisper-tiny-hf/model.safetensors"
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("whisper-tiny model not available")
	}

	cfg := Tiny()
	enc, dec, err := LoadModel(modelPath, cfg)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	// Use synthetic mel input
	T := 100
	melFlat := make([]float32, cfg.NumMelBins*T)
	for i := range melFlat {
		melFlat[i] = float32(i%80) * 0.01
	}

	encoderOutput := enc.Forward(melFlat, T)
	encLen := len(encoderOutput) / cfg.EncoderDModel
	state := NewDecoderState(cfg, encoderOutput, encLen, dec)

	lang, confidence := DetectLanguage(dec, state)
	t.Logf("Detected language: %s (confidence: %.3f)", lang, confidence)

	// Verify it returns a valid language code
	if _, ok := LanguageTokens[lang]; !ok {
		t.Fatalf("invalid language code: %q", lang)
	}
	if confidence < 0 {
		t.Fatalf("confidence should be non-negative: %f", confidence)
	}
}

func TestTranscribeWithLanguageDetectSmoke(t *testing.T) {
	cfg := Tiny()
	cfg.EncoderLayers = 0
	cfg.DecoderLayers = 0
	cfg.MaxDecoderLength = 2
	enc, dec := buildZeroWeightModel(t, cfg)
	w := &Whisper{Encoder: enc, Decoder: dec, Config: cfg}
	text, lang, err := w.TranscribeWithLanguageDetect(make([]float32, 16000))
	if err != nil {
		t.Fatalf("TranscribeWithLanguageDetect: %v", err)
	}
	if _, ok := LanguageTokens[lang]; !ok {
		t.Fatalf("invalid detected language %q", lang)
	}
	t.Logf("language-detect smoke lang=%s text=%q", lang, text)
}

func TestLanguageTokensComplete(t *testing.T) {
	// Verify token map is non-empty and bidirectional
	if len(LanguageTokens) < 50 {
		t.Fatalf("expected at least 50 languages, got %d", len(LanguageTokens))
	}
	if len(LanguageNames) != len(LanguageTokens) {
		t.Fatalf("LanguageNames size %d != LanguageTokens size %d", len(LanguageNames), len(LanguageTokens))
	}
	// Verify English is present
	if tok, ok := LanguageTokens["en"]; !ok || tok != TokenEnglish {
		t.Fatalf("English token mismatch: ok=%v tok=%d", ok, tok)
	}
}
