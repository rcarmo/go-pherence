package whisper

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio/media"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// TestWhisperTinyJFKWordAlignment is an opt-in trained-model check. The pinned
// model and fixture are intentionally not repository test dependencies.
func TestWhisperTinyJFKWordAlignment(t *testing.T) {
	modelDir := os.Getenv("WHISPER_TINY_MODEL_DIR")
	wav := os.Getenv("WHISPER_JFK_WAV")
	if modelDir == "" || wav == "" {
		t.Skip("set WHISPER_TINY_MODEL_DIR and WHISPER_JFK_WAV")
	}
	for _, name := range []string{"config.json", "generation_config.json", "tokenizer.json", "model.safetensors"} {
		if _, err := os.Stat(filepath.Join(modelDir, name)); err != nil {
			t.Skip(err)
		}
	}
	if _, err := os.Stat(wav); err != nil {
		t.Skip(err)
	}

	tok, err := LoadTokenizer(filepath.Join(modelDir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	modelJSON, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	generationJSON, err := os.ReadFile(filepath.Join(modelDir, "generation_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	source, err := safetensors.Open(filepath.Join(modelDir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	model, generation, err := LoadConfiguredModelSourceChecked(context.Background(), source, modelJSON, generationJSON, tok)
	if err != nil {
		t.Fatal(err)
	}
	reader, err := media.OpenCanonicalPCM(context.Background(), wav)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	pcm := make([]float32, reader.Timeline().Samples)
	if n, err := reader.ReadSamplesAt(context.Background(), pcm, 0); err != nil || n != len(pcm) {
		t.Fatal(n, err)
	}
	padded := make([]float32, model.Config.MaxLength*160)
	copy(padded, pcm)
	mel, frames, err := MelFlatFromSamplesCheckedContext(context.Background(), padded, model.Config)
	if err != nil {
		t.Fatal(err)
	}
	encoderOutput := model.Encoder.Forward(mel, frames)
	state := NewDecoderState(model.Config, encoderOutput, (frames+1)/2, model.Decoder)

	// Pinned tokenizer IDs for the checked JFK transcript. The aligner accepts
	// generated text tokens only; EOT and prompt/control tokens are excluded.
	tokens := []int{400, 370, 452, 7177, 6280, 1029, 406, 437, 428, 1941, 393, 360, 337, 291, 1029, 437, 291, 393, 360, 337, 428, 1941, 13}
	words, err := AlignWordsChecked(context.Background(), model.Decoder, state, tok, generation, "en", tokens, (len(pcm)+159)/160)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"And", "so", "my", "fellow", "Americans", "ask", "not", "what", "your", "country", "can", "do", "for", "you", "ask", "what", "you", "can", "do", "for", "your", "country."}
	got := make([]string, len(words))
	for i, word := range words {
		got[i] = word.Word
		if word.Start < 0 || word.End < word.Start || word.End > float64(len(pcm))/16000 || i > 0 && word.Start < words[i-1].End {
			t.Fatalf("word %d has invalid timing: %+v", i, word)
		}
	}
	wantStarts := []float64{0, 1.08, 1.34, 1.70, 2.28, 3.80, 4.70, 5.70, 5.96, 6.42, 6.72, 6.96, 7.24, 8.14, 8.62, 8.98, 9.22, 9.44, 9.70, 9.88, 10.10, 10.60}
	if !reflect.DeepEqual(got, want) || len(words) != len(wantStarts) {
		t.Fatalf("words=%+v", words)
	}
	for i := range words {
		if delta := words[i].Start - wantStarts[i]; delta < -0.02 || delta > 0.02 {
			t.Fatalf("word %d start=%.2f want %.2f (+/-0.02): %+v", i, words[i].Start, wantStarts[i], words)
		}
	}
	if delta := words[len(words)-1].End - 10.98; delta < -0.02 || delta > 0.02 {
		t.Fatalf("last word end=%.2f want 10.98 (+/-0.02): %+v", words[len(words)-1].End, words[len(words)-1])
	}
	t.Logf("WHISPER_WORD_ALIGNMENT words=%d first=%.2f last=%.2f", len(words), words[0].Start, words[len(words)-1].End)
}
