package gliner2

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func TestMixedSchemaUpstreamRouting(t *testing.T) {
	path := os.Getenv("GLINER_TOKENIZER_JSON")
	if path == "" {
		t.Skip("set GLINER_TOKENIZER_JSON")
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
	var ref struct {
		Text           string
		Schemas        []TextSchema
		IDs, Positions []int
		Groups         [][]int
	}
	raw, err := os.ReadFile("testdata/mixed_schema_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	got, err := tok.PrepareSchemas(ref.Text, ref.Schemas, 4096)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.IDs, ref.IDs) || !reflect.DeepEqual(got.TextPositions, ref.Positions) {
		t.Fatal("combined sequence mismatch")
	}
	for i, g := range got.Groups {
		if !reflect.DeepEqual(g.QueryPositions, ref.Groups[i]) {
			t.Fatal("query positions mismatch", i)
		}
	}
	if _, err = tok.PrepareSchemas(ref.Text, ref.Schemas, len(ref.IDs)-1); err == nil {
		t.Fatal("token budget ignored")
	}
	single, err := tok.PrepareSchemas(ref.Text, ref.Schemas[:1], 4096)
	if err != nil {
		t.Fatal(err)
	}
	base, err := tok.PrepareTextSchema(ref.Text, ref.Schemas[0], 4096)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(single.IDs, base.IDs) || !reflect.DeepEqual(single.TextPositions, base.TextPositions) {
		t.Fatal("single schema regression")
	}
}
