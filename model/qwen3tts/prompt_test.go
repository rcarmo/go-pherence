package qwen3tts

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestBuildCustomVoicePrompt(t *testing.T) {
	tok := &tokenizer.Tokenizer{Vocab: map[string]int{"Hello": 9707, "Ġworld": 1879}}
	prompt, err := BuildCustomVoicePrompt(tok, "Hello world", Ryan, English)
	if err != nil {
		t.Fatal(err)
	}
	wantText := []uint32{IMStart, Assistant, Newline, TTSPad, TTSPad, TTSPad, TTSPad, TTSPad, TTSBOS, 9707, 1879}
	wantCodec := []uint32{CodecThink, CodecThinkBOS, 2050, CodecThinkEOS, 3061, CodecPad, CodecBOS}
	if !eq(prompt.Text, wantText) {
		t.Fatalf("text=%v", prompt.Text)
	}
	if !eq(prompt.Codec, wantCodec) {
		t.Fatalf("codec=%v", prompt.Codec)
	}
}

func TestBuildCustomVoicePromptRejectsEmptyText(t *testing.T) {
	tok := &tokenizer.Tokenizer{Vocab: map[string]int{"Hello": 9707}}
	_, err := BuildCustomVoicePrompt(tok, "", Ryan, English)
	if err == nil {
		t.Fatal("expected empty tokenization error")
	}
}

func TestLoadTokenizerFallbackVocabMerges(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "vocab.json"), []byte(`{"H":1,"e":2,"He":3}`), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "merges.txt"), []byte("#version: 0.2\nH e\n"), 0644); err != nil {
		t.Fatal(err)
	}
	tok, err := LoadTokenizer(dir)
	if err != nil {
		t.Fatal(err)
	}
	if tok.VocabSize() != 3 || len(tok.Merges) != 1 || tok.Merges[0] != [2]string{"H", "e"} {
		t.Fatalf("tokenizer=%+v merges=%v", tok, tok.Merges)
	}
}
