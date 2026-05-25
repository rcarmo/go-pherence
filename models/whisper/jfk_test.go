package whisper

import (
	"os"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/loader/audio"
)

func TestTranscribeJFK(t *testing.T) {
	modelPath := "../../models/whisper-tiny-hf/model.safetensors"
	tokPath := "../../models/whisper-tiny-hf/tokenizer.json"
	audioPath := "../../testdata/jfk.wav"
	for _, p := range []string{modelPath, tokPath, audioPath} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("missing: %s", p)
		}
	}

	if err := LoadTokenizerGlobal(tokPath); err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	defer func() { DefaultTokenizer = nil }()

	cfg := Tiny()
	enc, dec, err := LoadModel(modelPath, cfg)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	samples, rate, err := audio.WAV(audioPath)
	if err != nil {
		t.Fatalf("WAV: %v", err)
	}
	if rate != 16000 {
		samples = audio.ResampleSinc(samples, rate, 16000)
	}
	audioDur := float64(len(samples)) / 16000
	t.Logf("JFK audio: %.1fs", audioDur)

	w := &Whisper{Encoder: enc, Decoder: dec, Config: cfg}

	start := time.Now()
	text, err := w.TranscribeFromSamples(samples)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}

	t.Logf("Time: %v (RTF=%.2f)", elapsed, elapsed.Seconds()/audioDur)
	t.Logf("Transcription:\n%s", text)
}
