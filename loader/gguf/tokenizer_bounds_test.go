package gguf

import "testing"

func TestTokenizerOutputBounds(t *testing.T) {
	if _, err := NewTokenizer(nil); err == nil {
		t.Fatal("nil GGUF accepted")
	}
	tok := &Tokenizer{vocab: []string{"a", "b"}}
	ids, err := tok.parseEncodedOutput("# log\n0 -> a\n1 -> b\n")
	if err != nil || len(ids) != 2 || ids[1] != 1 {
		t.Fatal(ids, err)
	}
	for _, s := range []string{"-1 -> bad", "2 -> bad", "no tokens"} {
		if _, err := tok.parseEncodedOutput(s); err == nil {
			t.Fatal("invalid tokenizer output accepted")
		}
	}
}
