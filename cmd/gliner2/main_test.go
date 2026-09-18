package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestUsageAndValidation(t *testing.T) {
	for _, tc := range []struct {
		args []string
		ok   bool
	}{{[]string{"-help"}, true}, {nil, false}, {[]string{"-model", "none", "-text", "x", "-label", "person", "-threshold", "2"}, false}} {
		var out, errout bytes.Buffer
		err := run(tc.args, &out, &errout)
		if (err == nil) != tc.ok {
			t.Fatalf("%v: %v", tc.args, err)
		}
	}
}

func TestRecordSpec(t *testing.T) {
	s, err := recordSpec([]string{"name:str", "city:list"}, "natural", "")
	if err != nil || s.AnchorQueryID != 0 || !s.Fields[0].Scalar || s.Fields[1].Scalar {
		t.Fatal(s, err)
	}
	for _, fields := range [][]string{nil, {"name:required"}, {"name:str", "name:list"}, {"name"}} {
		if _, err := recordSpec(fields, "natural", ""); err == nil {
			t.Fatal("accepted", fields)
		}
	}
	if _, err := recordSpec([]string{"name:str"}, "latent", "name"); err == nil {
		t.Fatal("latent anchor accepted")
	}
	if _, err := recordSpec([]string{"name:str"}, "natural", "missing"); err == nil {
		t.Fatal("missing anchor accepted")
	}
}

func TestRecordFieldModifiers(t *testing.T) {
	s, err := recordSpec([]string{"name:str,required,exclusive", "city:list,exclusive"}, "natural", "")
	if err != nil || !s.Fields[0].Required || !s.Fields[0].Exclusive || !s.Fields[1].Exclusive {
		t.Fatal(s, err)
	}
	for _, v := range []string{"name:str,unknown", "name:str,exclusive,exclusive", "name:str,"} {
		if _, err := recordSpec([]string{v}, "natural", ""); err == nil {
			t.Fatal("accepted", v)
		}
	}
}

func TestRequiredListModifier(t *testing.T) {
	s, err := recordSpec([]string{"places:list,required,exclusive"}, "anchorless", "")
	if err != nil || !s.Fields[0].Required || !s.Fields[0].Exclusive || s.Fields[0].Scalar {
		t.Fatal(s, err)
	}
}

func TestTextSchemaFileValidation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "schema.json")
	for _, tc := range []struct {
		raw string
		ok  bool
	}{
		{`{"parent":"entities","marker":"[E]","labels":["person"],"descriptions":[{"label":"person","text":"A human"}]}`, true},
		{`{"parent":"entities","marker":"[E]","labels":["x","x"]}`, false},
		{`{"parent":"r","marker":"[R]","labels":["head"]}`, false},
		{`{"parent":"e","marker":"[E]","labels":["x"],"typo":1}`, false},
		{`{"parent":"e","marker":"[E]","labels":["x"]} {}`, false},
	} {
		if err := os.WriteFile(path, []byte(tc.raw), 0600); err != nil {
			t.Fatal(err)
		}
		_, err := readTextSchema(path)
		if (err == nil) != tc.ok {
			t.Fatalf("%s: %v", tc.raw, err)
		}
	}
	var out bytes.Buffer
	if err := run([]string{"-model", "none", "-text", "x", "-schema", path, "-label", "person"}, &out, &out); err == nil {
		t.Fatal("mixed flags accepted")
	}
}
