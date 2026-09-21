package qev

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestCompileSchemaStableEnumBooleanContract(t *testing.T) {
	compiled, err := CompileSchema([]byte(`{
		"urgent":{"type":"boolean","description":"Needs urgent handling?"},
		"category":{"type":"enum","choices":["billing","technical","other"],"description":"Request type"}
	}`), "Use the request state.")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(compiled.Fields), 2; got != want {
		t.Fatalf("fields=%d want %d", got, want)
	}
	// Field order is lexical rather than Go map iteration order.
	if compiled.Fields[0].Name != "category" || compiled.Fields[1].Name != "urgent" {
		t.Fatalf("field order=%q,%q", compiled.Fields[0].Name, compiled.Fields[1].Name)
	}
	if got := compiled.Fields[0].Candidates[1].ID; got != "category:001" {
		t.Fatalf("candidate ID=%q", got)
	}
	if got := compiled.Inputs[0].Suffix; got != `  "category": "` {
		t.Fatalf("suffix=%q", got)
	}
	if got := compiled.Inputs[0].Candidates[0].Encoded; got != "billing\"" {
		t.Fatalf("candidate remainder=%q", got)
	}
	if !strings.Contains(compiled.SystemText, `"urgent": Needs urgent handling?`) || !strings.HasSuffix(compiled.SystemText, "Use the request state.") {
		t.Fatalf("system text=%q", compiled.SystemText)
	}
}

func TestCompileJSONSchemaEnumBoolean(t *testing.T) {
	compiled, err := CompileSchema([]byte(`{
		"type":"object",
		"properties":{
			"active":{"type":"boolean"},
			"color":{"type":"string","enum":["red","blue"]}
		}
	}`), "")
	if err != nil {
		t.Fatal(err)
	}
	if len(compiled.Fields) != 2 || compiled.Fields[0].Name != "active" || compiled.Fields[1].Name != "color" {
		t.Fatalf("fields=%+v", compiled.Fields)
	}
}

func TestCompileSchemaRejectsOutOfScopeAndMalformedFields(t *testing.T) {
	tests := []struct {
		name   string
		schema string
		want   string
	}{
		{"not object", `[]`, "object"},
		{"empty", `{}`, "1-32"},
		{"compact description", `{"x":{"type":"boolean"}}`, "description"},
		{"unsupported", `{"x":{"type":"integer","description":"x"}}`, "boolean and enum"},
		{"non string enum", `{"x":{"type":"enum","description":"x","choices":[1]}}`, "must be strings"},
		{"duplicate", `{"x":{"type":"enum","description":"x","choices":["a","a"]}}`, "duplicate"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := CompileSchema([]byte(tc.schema), "")
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error=%v want substring %q", err, tc.want)
			}
		})
	}
}

func TestRequestNormalizeAndValidate(t *testing.T) {
	r := Request{Schema: json.RawMessage(`{"x":{"type":"boolean","description":"x"}}`), Contexts: []string{"one"}}
	if err := r.NormalizeAndValidate(); err != nil {
		t.Fatal(err)
	}
	if r.Mode != ModeAuto || r.TreeMax != DefaultTreeMax || !r.AllowCache() {
		t.Fatalf("normalized request=%+v", r)
	}
	no := false
	r.CachePrompt = &no
	if r.AllowCache() {
		t.Fatal("cache_prompt=false ignored")
	}
	for _, bad := range []Request{
		{Schema: r.Schema},
		{Schema: r.Schema, Contexts: []string{""}},
		{Schema: r.Schema, Contexts: []string{"x"}, Mode: "greedy"},
		{Schema: r.Schema, Contexts: []string{"x"}, TreeMax: MaxCandidates + 1},
	} {
		if err := bad.NormalizeAndValidate(); err == nil {
			t.Fatalf("accepted bad request %+v", bad)
		}
	}
}
