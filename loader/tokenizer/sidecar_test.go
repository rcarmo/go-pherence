package tokenizer

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestSidecarOrdinaryAddedTokenIsAtomicNotSpecial(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{"model":{"type":"BPE","vocab":{"x":0},"merges":[]}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	config := `{"added_tokens_decoder":{"1":{"content":"<think>","special":false,"normalized":false},"2":{"content":"<|start|>","special":true,"normalized":false}}}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	tok, err := LoadWithConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got := tok.Encode("<|start|><think>x"); !reflect.DeepEqual(got, []int{2, 1, 0}) {
		t.Fatal(got)
	}
	if _, ok := tok.AddedSpecial["<think>"]; ok {
		t.Fatal("ordinary token promoted to special")
	}
	if tok.Decode([]int{1}) != "<think>" {
		t.Fatal("decode mismatch")
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{"added_tokens_decoder":{"1":{"content":"<think>","lstrip":true}}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadWithConfig(dir); err == nil {
		t.Fatal("unsupported policy accepted")
	}
}
