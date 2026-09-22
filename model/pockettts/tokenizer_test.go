package pockettts

import (
	"os"
	"reflect"
	"testing"
)

func releasedTokenizer(t *testing.T) *Tokenizer {
	t.Helper()
	path := releasedTokenizerPath(t)
	file, err := os.Open(path)
	if err != nil {
		t.Skipf("released Pocket TTS tokenizer unavailable: %v", err)
	}
	defer file.Close()
	tok, err := LoadTokenizer(file)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func TestReleasedPocketTokenizerParity(t *testing.T) {
	tok := releasedTokenizer(t)
	if tok.VocabSize() != 4000 {
		t.Fatalf("vocab=%d", tok.VocabSize())
	}
	for _, test := range []struct {
		text string
		want []uint32
	}{
		{"Hello world!", []uint32{2994, 578, 682}},
		{" hello  world ", []uint32{260, 1876, 393, 260, 578, 260}},
		{"café", []uint32{331, 2250, 745}},
		{"🙂", []uint32{260, 244, 163, 157, 134}},
		{"naïve\ntext", []uint32{913, 199, 179, 314, 14, 274, 1838, 274}},
		{"", []uint32{}},
		{"Português — ação.", []uint32{3764, 483, 1118, 3131, 261, 260, 3133, 267, 2467, 2522, 263}},
	} {
		got, err := tok.Encode(test.text)
		if err != nil {
			t.Fatalf("Encode(%q): %v", test.text, err)
		}
		if !reflect.DeepEqual(got, test.want) {
			t.Fatalf("Encode(%q)=%v want=%v", test.text, got, test.want)
		}
	}
}

func TestPocketTokenizerRejectsInvalidUTF8(t *testing.T) {
	tok := releasedTokenizer(t)
	if _, err := tok.Encode(string([]byte{0xff})); err == nil {
		t.Fatal("accepted invalid UTF-8")
	}
}
