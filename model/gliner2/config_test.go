package gliner2

import (
	"bytes"
	"encoding/json"
	"os"
	"testing"
)

func TestPublishedBoundaryConfig(t *testing.T) {
	raw, err := os.ReadFile("testdata/config.json")
	if err != nil {
		t.Fatal(err)
	}
	c, err := LoadConfig(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	if c.ModelName != "microsoft/deberta-v3-base" || c.Architecture != "boundary" {
		t.Fatalf("unexpected model %s %s", c.ModelName, c.Architecture)
	}
	if len(c.Raw) == 0 {
		t.Fatal("raw contract not preserved")
	}
	var obj map[string]any
	if err = json.Unmarshal(raw, &obj); err != nil {
		t.Fatal(err)
	}
	obj["architecture"] = "span"
	bad, _ := json.Marshal(obj)
	if _, err = LoadConfig(bytes.NewReader(bad)); err == nil {
		t.Fatal("accepted unsupported span architecture")
	}
	if _, err = LoadConfig(bytes.NewReader([]byte("{}"))); err == nil {
		t.Fatal("accepted missing config")
	}
	if _, err = LoadConfig(bytes.NewReader(append(raw, []byte(" {}")...))); err == nil {
		t.Fatal("accepted trailing data")
	}
}
