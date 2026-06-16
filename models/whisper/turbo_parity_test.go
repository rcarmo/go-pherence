package whisper

import (
	"os"
	"strings"
	"testing"
	"unicode"

	"github.com/rcarmo/go-pherence/loader/audio"
)

func TestLargeV3TurboJFKCPUTranscriptParity(t *testing.T) {
	modelPath := "../../models/whisper-large-v3-turbo-hf/model.safetensors"
	tokPath := "../../models/whisper-large-v3-turbo-hf/tokenizer.json"
	audioPath := "../../testdata/jfk.wav"
	requireAssets := os.Getenv("WHISPER_REQUIRE_TURBO_PARITY") == "1"
	for _, p := range []string{modelPath, tokPath, audioPath} {
		if _, err := os.Stat(p); err != nil {
			if requireAssets {
				t.Fatalf("required Whisper turbo parity asset missing: %s", p)
			}
			t.Skipf("missing: %s", p)
		}
	}
	if err := LoadTokenizerGlobal(tokPath); err != nil {
		t.Fatalf("LoadTokenizerGlobal: %v", err)
	}
	defer func() { DefaultTokenizer = nil }()

	cfg := LargeV3Turbo()
	cfg.MaxDecoderLength = 80
	enc, dec, err := LoadModel(modelPath, cfg)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	if !UseGPULMHead() && dec.lmHeadGPU != nil {
		t.Fatalf("LoadModel enabled GPU LM-head without GO_PHERENCE_WHISPER_GPU_LM_HEAD or GPU graph flag")
	}
	samples, sr, err := audio.WAV(audioPath)
	if err != nil {
		t.Fatalf("WAV: %v", err)
	}
	if sr != 16000 {
		samples = audio.ResampleSinc(samples, sr, 16000)
	}
	w := &Whisper{Encoder: enc, Decoder: dec, Config: cfg}
	text, err := w.TranscribeFromSamplesPrompt(samples, TokenEnglish, TokenTranscribe)
	if err != nil {
		t.Fatalf("TranscribeFromSamplesPrompt: %v", err)
	}
	got := normalizeTranscript(text)
	want := "and so my fellow americans ask not what your country can do for you ask what you can do"
	if got != want {
		t.Fatalf("normalized transcript mismatch\ngot:  %q\nwant: %q\nraw:  %q", got, want, text)
	}
}

func normalizeTranscript(s string) string {
	s = strings.ToLower(s)
	var b strings.Builder
	lastSpace := true
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
			lastSpace = false
			continue
		}
		if !lastSpace {
			b.WriteByte(' ')
			lastSpace = true
		}
	}
	return strings.TrimSpace(b.String())
}
