package mosstranscribe

import (
	"os"
	"testing"

	basetokenizer "github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestProcessorEncodePrompt(t *testing.T) {
	vocab := map[string]int{"A": 1, "B": 2, AudioPadToken: 99}
	for digit := 0; digit < 10; digit++ {
		vocab[string(rune('0'+digit))] = 10 + digit
	}
	tok := &basetokenizer.Tokenizer{
		Vocab:        vocab,
		InvVocab:     map[int]string{},
		AddedSpecial: map[string]int{AudioPadToken: 99},
	}
	processor, err := NewProcessor(tok, 99)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := processor.EncodePrompt("A"+AudioPadToken+"B", 1, 8)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{1, 99, 2}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("ids=%v want %v", ids, want)
		}
	}
	if _, err := processor.EncodePrompt("no placeholder", 1, 8); err == nil {
		t.Fatal("accepted missing placeholder")
	}
	if _, err := processor.EncodePrompt(AudioPadToken, 20, 4); err == nil {
		t.Fatal("accepted sequence beyond max length")
	}
}

func TestPinnedMOSSRealTokenizer(t *testing.T) {
	modelDir := os.Getenv("MOSS_TRANSCRIBE_MODEL_DIR")
	if modelDir == "" {
		t.Skip("set MOSS_TRANSCRIBE_MODEL_DIR for tokenizer parity")
	}
	processor, err := LoadProcessor(modelDir, 151671)
	if err != nil {
		t.Fatal(err)
	}
	prompt := "<|im_start|>user\nTranscribe. " + AudioPadToken + "<|im_end|>\n<|im_start|>assistant\n"
	ids, err := processor.EncodePrompt(prompt, 1, 131072)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{151644, 872, 198, 3167, 3114, 13, 220, 151671, 151645, 198, 151644, 77091, 198}
	if len(ids) != len(want) {
		t.Fatalf("ids=%v want %v", ids, want)
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("ids=%v want %v", ids, want)
		}
	}
}
