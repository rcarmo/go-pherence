package gliner2

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func TestPublishedTokenizerParity(t *testing.T) {
	path := os.Getenv("GLINER_TOKENIZER_JSON")
	if path == "" {
		t.Skip("set GLINER_TOKENIZER_JSON to published tokenizer.json")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	tok, err := LoadTokenizer(f)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("testdata/tokenizer_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var examples []struct {
		Text string
		IDs  []int
	}
	if err = json.Unmarshal(raw, &examples); err != nil {
		t.Fatal(err)
	}
	for _, ex := range examples {
		got, err := tok.Encode(ex.Text, true)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, ex.IDs) {
			t.Errorf("%q got %v want %v", ex.Text, got, ex.IDs)
		}
	}
}

func TestSmallTokenizerUpstreamParity(t *testing.T) {
	f, err := os.Open("testdata/tokenizer_small.json")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	tok, err := LoadTokenizer(f)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("testdata/tokenizer_small_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var examples []struct {
		Text string
		IDs  []int
	}
	if err = json.Unmarshal(raw, &examples); err != nil {
		t.Fatal(err)
	}
	for _, ex := range examples {
		got, err := tok.Encode(ex.Text, true)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, ex.IDs) {
			t.Errorf("%q got %v want %v", ex.Text, got, ex.IDs)
		}
	}
}

func TestTokenizerRejectsTrailingAndDuplicateSpecials(t *testing.T) {
	raw, err := os.ReadFile("testdata/tokenizer_small.json")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := LoadTokenizer(bytes.NewReader(append(append([]byte(nil), raw...), []byte(" {}")...))); err == nil {
		t.Fatal("trailing JSON accepted")
	}
	var obj map[string]any
	if err = json.Unmarshal(raw, &obj); err != nil {
		t.Fatal(err)
	}
	added := obj["added_tokens"].([]any)
	obj["added_tokens"] = append(added, added[0])
	bad, _ := json.Marshal(obj)
	if _, err := LoadTokenizer(bytes.NewReader(bad)); err == nil {
		t.Fatal("duplicate special accepted")
	}
}
