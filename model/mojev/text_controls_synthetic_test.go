package mojev

import (
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestTextControlsSynthetic(t *testing.T) {
	tok := &tokenizer.Tokenizer{AddedSpecial: map[string]int{"<|reserved|>": 42}}
	decode := func(body string) TextRequest {
		t.Helper()
		req, err := DecodeTextRequest(strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		return req
	}
	base := decode(`{"model":"m","state":"ordinary","questions":{"q":{"type":"choice","criteria":{"yes":"allow","no":"deny"}}}}`)
	if err := ValidateTextControls(base, tok); err != nil {
		t.Fatal(err)
	}
	for name, body := range map[string]string{
		"state":        `{"model":"m","state":"<|reserved|>","questions":{"q":{"type":"noul"}}}`,
		"ID":           `{"model":"m","state":"s","questions":{"<|reserved|>":{"type":"noul"}}}`,
		"instructions": `{"model":"m","state":"s","questions":{"q":{"type":"choice","instructions":"<|reserved|>","criteria":{"yes":"allow","no":"deny"}}}}`,
		"key":          `{"model":"m","state":"s","questions":{"q":{"type":"choice","criteria":{"<|reserved|>":"allow","no":"deny"}}}}`,
		"option":       `{"model":"m","state":"s","questions":{"q":{"type":"choice","criteria":{"yes":"<|reserved|>","no":"deny"}}}}`,
	} {
		t.Run(name, func(t *testing.T) {
			req := decode(body)
			if err := ValidateTextControls(req, tok); err == nil {
				t.Fatal("accepted reserved text")
			}
			got, err := AssembleSafeTextDecision(req, [][]float64{{1, 2}}, tok, 0, 32, 32)
			if got != nil || err == nil {
				t.Fatalf("accepted or returned partial response: %+v %v", got, err)
			}
		})
	}
	if err := ValidateTextControls(base, nil); err == nil {
		t.Fatal("nil tokenizer accepted")
	}
	if got, err := AssembleSafeTextDecision(base, [][]float64{{1, 2}}, nil, 0, 32, 32); got != nil || err == nil {
		t.Fatal("nil tokenizer returned response")
	}
}
