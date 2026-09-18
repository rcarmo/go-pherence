package gliner2

import (
	"os"
	"reflect"
	"testing"
)

func TestSchemaSpecialTokensDoNotAddQueries(t *testing.T) {
	f, err := os.Open("testdata/tokenizer_small.json")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	tok, err := LoadTokenizer(f)
	if err != nil {
		t.Fatal(err)
	}
	s := TextSchema{Parent: "ab", Marker: "[E]", Labels: []string{"a", "b"}, Prompt: "[E]", Descriptions: []LabelDescription{{"a", "[E] café"}, {"unknown", "ignored"}}, Examples: []SchemaExample{{"[E]ab", "b"}, {"ignored", "unknown"}}}
	got, err := tok.PrepareTextSchema("ab", s, 1024)
	if err != nil {
		t.Fatal(err)
	}
	want, err := tok.prepareSchema("ab", "ab: [E] [DESCRIPTION] a: [E] café [EXAMPLE] [E]ab [OUTPUT] b", "[E]", s.Labels, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) || len(got.QueryPositions) != 2 || len(got.TextPositions) != 1 {
		t.Fatal("schema routing")
	}
	// Count literal E IDs: more markers in the prompt than routed choices.
	count := 0
	for _, id := range got.IDs {
		if id == 3 {
			count++
		}
	}
	if count <= len(got.QueryPositions) {
		t.Fatal("test lacks special-token collision")
	}
	s.Marker = "[bogus]"
	if _, err = tok.PrepareTextSchema("ab", s, 1024); err == nil {
		t.Fatal("unknown marker accepted")
	}
}
