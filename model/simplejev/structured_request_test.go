package simplejev

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestPinnedStructuredStateRequestFixture(t *testing.T) {
	if err := verifyFixtureSHA("../../scripts/simplejev_oracle_structured_request.py", "04c40e3233e443ce15368397997694f21fc367a691fb3a7dea1538c27f7b6e9a"); err != nil {
		t.Fatal(err)
	}
	if err := verifyFixtureSHA("testdata/upstream_structured_request_v1.json", "4c328a1fd50190f7bcb361e24827dee7614a59541896c8c48ce209304ff9a1ca"); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("testdata/upstream_structured_request_v1.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema     int    `json:"schema"`
		Revision   string `json:"revision"`
		SchemaSHA  string `json:"request_schema_sha256"`
		BuilderSHA string `json:"prompt_builder_sha256"`
		Cases      []struct {
			Name      string          `json:"name"`
			Request   json.RawMessage `json:"request"`
			Accepted  bool            `json:"accepted"`
			Model     string          `json:"model"`
			Canonical string          `json:"canonical_state"`
			RawLogits bool            `json:"raw_logits"`
			IDs       []string        `json:"question_ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != 1 || fixture.Revision != "b02aa81c915a8193759b3cd33fef74721d6e005b" || fixture.SchemaSHA != "6fa1c1215e8fc9de7aedbee77db6520bbb811922df9c45858bf78c4e4a02ceac" || fixture.BuilderSHA != "14a885d3bfa44b0c8aa0ffc9ce84611a459957d938e1847f96d93c7f1840fa3d" || len(fixture.Cases) != 9 {
		t.Fatal("unexpected structured request fixture provenance")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			got, err := DecodeStructuredStateRequest(strings.NewReader(string(tc.Request)))
			if !tc.Accepted {
				if err == nil || len(got.Questions) != 0 {
					t.Fatalf("accepted rejected case: %+v err=%v", got, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			ids := make([]string, len(got.Questions))
			for i, q := range got.Questions {
				ids[i] = q.ID
			}
			if got.Model != tc.Model || got.StateJSON != tc.Canonical || got.RawLogits != tc.RawLogits || !reflect.DeepEqual(ids, tc.IDs) {
				t.Fatalf("request=%+v want state=%q ids=%v", got, tc.Canonical, tc.IDs)
			}
			if len(got.Questions) != 3 || len(got.Questions[0].Choices) != 2 || got.Questions[0].Choices[0].ID != "first" || got.Questions[1].Levels[1] != nil || len(got.Questions[2].NoulCriteria) != 1 {
				t.Fatalf("question order/null changed: %+v", got.Questions)
			}
		})
	}
}

func TestStructuredStateRequestRejectsUnsupported(t *testing.T) {
	base := `{"model":"fixture","state":{"a":1},"questions":{"q":{"type":"noul","instructions":""}}}`
	for name, body := range map[string]string{
		"nil":              "",
		"text":             strings.Replace(base, `"state":{"a":1}`, `"state":"plain"`, 1),
		"duplicate state":  strings.Replace(base, `"state":{"a":1}`, `"state":{"a":1},"state":[]`, 1),
		"duplicate nested": strings.Replace(base, `"state":{"a":1}`, `"state":{"a":1,"a":2}`, 1),
		"fraction":         strings.Replace(base, `"state":{"a":1}`, `"state":{"a":1.5}`, 1),
		"nonfinite":        strings.Replace(base, `"state":{"a":1}`, `"state":{"a":NaN}`, 1),
		"invalid escape":   strings.Replace(base, `"state":{"a":1}`, `"state":{"a":"\ud800"}`, 1),
		"messages":         strings.Replace(base, `"questions":`, `"messages":[{"role":"user","content":"hi"}],"questions":`, 1),
		"media":            strings.Replace(base, `"questions":`, `"media_io_kwargs":{},"questions":`, 1),
		"trailing":         base + ` {}`,
		"invalid UTF8":     base + "\xff",
	} {
		t.Run(name, func(t *testing.T) {
			if name == "nil" {
				if _, err := DecodeStructuredStateRequest(nil); err == nil {
					t.Fatal("accepted nil")
				}
				return
			}
			if got, err := DecodeStructuredStateRequest(strings.NewReader(body)); err == nil || len(got.Questions) != 0 {
				t.Fatalf("accepted %s: %+v err=%v", name, got, err)
			}
		})
	}
	large := fmt.Sprintf(`{"model":"fixture","state":{"a":"%s"},"questions":{"q":{"type":"noul","instructions":""}}}`, strings.Repeat("x", MaxStructuredStateBytes))
	if _, err := DecodeStructuredStateRequest(strings.NewReader(large)); err == nil {
		t.Fatal("accepted oversized structured state")
	}
	if _, err := DecodeStructuredStateRequest(strings.NewReader(strings.Repeat(" ", MaxRequestBytes+1))); err == nil {
		t.Fatal("accepted oversized request")
	}
	if _, err := DecodeStructuredStateRequest(failingRequestReader{}); !errors.Is(err, errStructuredRead) {
		t.Fatalf("reader failure not returned: %v", err)
	}
	if got, err := DecodeTextStateRequest(strings.NewReader(base)); err == nil || len(got.Questions) != 0 {
		t.Fatalf("text decoder accepted structured state: %+v %v", got, err)
	}
	input := []byte(base)
	got, err := DecodeStructuredStateRequest(bytes.NewReader(input))
	if err != nil {
		t.Fatal(err)
	}
	clear(input)
	if got.StateJSON != `{"a":1}` || len(got.Questions) != 1 || got.Questions[0].ID != "q" {
		t.Fatalf("returned request aliases input: %+v", got)
	}
}

var errStructuredRead = errors.New("structured request read failed")

type failingRequestReader struct{}

func (failingRequestReader) Read([]byte) (int, error) { return 0, errStructuredRead }
