package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestLoadTokenizerMissingVocabDoesNotPanic(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	json := `{"model":{"merges":null},"added_tokens":[{"id":7,"content":"<x>"}]}`
	if err := os.WriteFile(path, []byte(json), 0644); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	tok, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if tok.Vocab["<x>"] != 7 || tok.InvVocab[7] != "<x>" {
		t.Fatalf("added token not loaded: vocab=%v inv=%v", tok.Vocab, tok.InvVocab)
	}
}

func TestEncodePreservesAddedSpecialTokens(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	body := `{"model":{"vocab":{"a":1,"b":2},"merges":null},"added_tokens":[{"id":7,"content":"<x>","special":true}]}`
	if err := os.WriteFile(path, []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
	tok, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	got := tok.Encode("a<x>b<x>")
	want := []int{1, 7, 2, 7}
	if len(got) != len(want) {
		t.Fatalf("Encode=%v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Encode=%v want %v", got, want)
		}
	}
}

func TestLoadTokenizerRejectsMalformedMerges(t *testing.T) {
	dir := t.TempDir()
	cases := map[string]string{
		"string": `{"model":{"vocab":{"a":1,"b":2},"merges":["a b","broken"]}}`,
		"array":  `{"model":{"vocab":{"a":1,"b":2},"merges":[["a","b"],["", "c"]]}}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(dir, name+".json")
			if err := os.WriteFile(path, []byte(body), 0644); err != nil {
				t.Fatalf("write tokenizer: %v", err)
			}
			if _, err := Load(path); err == nil {
				t.Fatal("Load accepted malformed merges")
			}
		})
	}
}

func TestTokenizerNilSafe(t *testing.T) {
	var tok *Tokenizer
	if got := tok.Encode("hello"); got != nil {
		t.Fatalf("nil Encode=%v, want nil", got)
	}
	if got := tok.Decode([]int{1}); got != "" {
		t.Fatalf("nil Decode=%q, want empty", got)
	}
}

func TestDecodePreservesUnknownUnicodeRunes(t *testing.T) {
	tok := &Tokenizer{InvVocab: map[int]string{1: "☃"}}
	if got := tok.Decode([]int{1}); got != "☃" {
		t.Fatalf("Decode unicode=%q, want snowman", got)
	}
}

func TestTokenizerNilVocabSizeAndByteMaps(t *testing.T) {
	var tok *Tokenizer
	if got := tok.VocabSize(); got != 0 {
		t.Fatalf("nil tokenizer VocabSize=%d, want 0", got)
	}
	enc := getByteEncoder()
	dec := getByteDecoder()
	if len(enc) != 256 || len(dec) != 256 {
		t.Fatalf("byte maps sizes enc=%d dec=%d, want 256", len(enc), len(dec))
	}
	for b := 0; b < 256; b++ {
		r := enc[byte(b)]
		if got := dec[r]; got != byte(b) {
			t.Fatalf("byte map roundtrip %d -> %U -> %d", b, r, got)
		}
	}
}

func TestLoadTokenizerAppliesNFCNormalizer(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	body := `{"normalizer":{"type":"NFC"},"model":{"vocab":{"Ã©":1},"merges":null}}`
	if err := os.WriteFile(path, []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
	tok, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := tok.Encode("é"); len(got) != 1 || got[0] != 1 {
		t.Fatalf("Encode=%v, want [1]", got)
	}
}

func TestJSONMergeArraySizing(t *testing.T) {
	for _, input := range []string{
		`[]`, ` [ ] `, `["a b", "c d"]`, `[["a","b"],["c","d"]]`,
		`["a, b", "a\" b", "a\\ b", "[x] y"]`,
		`[null, true, 12, {"nested":[1,2]}, [[],{}]]`,
		"[\n\t[\"a\", \"b\"]\r\n, [\"c\", \"d\"]\n]",
	} {
		var items []json.RawMessage
		if err := json.Unmarshal([]byte(input), &items); err != nil {
			t.Fatal(err)
		}
		if got := jsonArrayLen([]byte(input)); got != len(items) {
			t.Fatalf("%s: count=%d want %d", input, got, len(items))
		}
	}
}

func FuzzJSONMergeArraySizing(f *testing.F) {
	for _, seed := range []string{`[]`, `[["a","b"],["c","d"]]`, `["a\\ b","c\" d"]`, `[null,{"a":[1,2]}]`} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, input string) {
		var items []json.RawMessage
		if json.Unmarshal([]byte(input), &items) != nil || strings.TrimSpace(input) == "null" {
			return
		}
		if got := jsonArrayLen([]byte(input)); got != len(items) {
			t.Fatalf("%q: count=%d want %d", input, got, len(items))
		}
	})
}

func TestLoadMergeRepresentations(t *testing.T) {
	want := [][2]string{{"a", "b"}, {"c", "d"}}
	for _, merges := range []string{`["a b", "c d"]`, ` [ ["a", "b"], ["c", "d"] ] `} {
		path := filepath.Join(t.TempDir(), "tokenizer.json")
		if err := os.WriteFile(path, []byte(`{"model":{"merges":`+merges+`}}`), 0600); err != nil {
			t.Fatal(err)
		}
		tok, err := Load(path)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(tok.Merges, want) {
			t.Fatalf("%s: got %v", merges, tok.Merges)
		}
	}
	for _, merges := range []string{`null`, `[]`, ` [ ] `} {
		path := filepath.Join(t.TempDir(), "tokenizer.json")
		if err := os.WriteFile(path, []byte(`{"model":{"merges":`+merges+`}}`), 0600); err != nil {
			t.Fatal(err)
		}
		tok, err := Load(path)
		if err != nil || len(tok.Merges) != 0 {
			t.Fatalf("%s: got %v %v", merges, tok, err)
		}
	}
	for _, merges := range []string{`{}`, `1`, `"a b"`, `[null,"a b"]`, `[["a","b"],"c d"]`, `["a b",["c","d"]]`} {
		path := filepath.Join(t.TempDir(), "tokenizer.json")
		if err := os.WriteFile(path, []byte(`{"model":{"merges":`+merges+`}}`), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := Load(path); err == nil {
			t.Fatalf("accepted invalid merges %s", merges)
		}
	}
}
