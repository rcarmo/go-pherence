package tokenizer

import (
	"encoding/json"
	"os"
	"testing"
)

func TestGemma4WhitespaceTokenizationPinnedLlamaCppFixture(t *testing.T) {
	dir := os.Getenv("GO_PHERENCE_GEMMA4_12B_TOKENIZER")
	if dir == "" {
		t.Skip("set GO_PHERENCE_GEMMA4_12B_TOKENIZER for pinned tokenizer parity")
	}
	tok, err := LoadWithConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("../../model/qev/testdata/llamacpp-gemma4-12b-boolean.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Inputs struct {
			SharedText  string `json:"shared_text"`
			ContextText string `json:"context_text"`
		} `json:"inputs"`
		Tokens struct {
			Shared    []int `json:"shared"`
			Context   []int `json:"context"`
			TruePath  []int `json:"true_path"`
			FalsePath []int `json:"false_path"`
		} `json:"tokens"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		text string
		want []int
	}{
		{fixture.Inputs.SharedText, fixture.Tokens.Shared[1:]},
		{fixture.Inputs.ContextText, fixture.Tokens.Context},
		{"  \"urgent\": true\n", fixture.Tokens.TruePath},
		{"  \"urgent\": false\n", fixture.Tokens.FalsePath},
	}
	for _, tc := range cases {
		got := tok.Encode(tc.text)
		if !sameTokenIDs(got, tc.want) {
			t.Fatalf("tokenize %q\ngot  %v\nwant %v", tc.text, got, tc.want)
		}
	}
}

func sameTokenIDs(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
