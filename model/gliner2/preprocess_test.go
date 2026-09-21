package gliner2

import (
	"reflect"
	"testing"
)

func TestSplitWordsOffsets(t *testing.T) {
	text := "İstanbul café Ada-Lovelace @User test@example.org!"
	words, err := SplitWords(text)
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, w := range words {
		got = append(got, w.Text)
		if w.ByteEnd <= w.ByteStart || w.End <= w.Start {
			t.Fatal(w)
		}
	}
	want := []string{"i\u0307stanbul", "café", "ada-lovelace", "@user", "test@example.org", "!"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q want %q", got, want)
	}
	if words[1].Start != 9 || text[words[1].ByteStart:words[1].ByteEnd] != "café" {
		t.Fatal(words[1])
	}
}
