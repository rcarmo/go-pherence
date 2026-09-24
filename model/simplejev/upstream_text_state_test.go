package simplejev

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
)

func TestPinnedUpstreamTextStateFixture(t *testing.T) {
	const scriptSHA = "ffd1880d752b660ff8f0deea5f430a0e536a28a218d74fb199ca7b6696bd15a3"
	const fixtureSHA = "bd95becb739cbffedc971682a70ae70dac8dd8c449c13719e2d00f9c6a714ee2"
	if err := verifyFixtureSHA("../../scripts/simplejev_oracle_text_state.py", scriptSHA); err != nil {
		t.Fatal(err)
	}
	if err := verifyFixtureSHA("testdata/upstream_text_state_v1.json", fixtureSHA); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("testdata/upstream_text_state_v1.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema         int    `json:"schema"`
		OracleRepo     string `json:"oracle_repo"`
		OracleRevision string `json:"oracle_revision"`
		SchemaSHA      string `json:"request_schema_sha256"`
		Cases          []struct {
			Name      string          `json:"name"`
			Request   json.RawMessage `json:"request"`
			Accepted  bool            `json:"accepted"`
			Model     string          `json:"model"`
			State     string          `json:"state"`
			RawLogits bool            `json:"raw_logits"`
			Questions []struct {
				ID           string          `json:"id"`
				Type         string          `json:"type"`
				Instructions *string         `json:"instructions"`
				Criteria     json.RawMessage `json:"criteria"`
			} `json:"questions"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != 1 || fixture.OracleRepo != "featherless-ai/simple-jev" || fixture.OracleRevision != "b02aa81c915a8193759b3cd33fef74721d6e005b" || fixture.SchemaSHA != "6fa1c1215e8fc9de7aedbee77db6520bbb811922df9c45858bf78c4e4a02ceac" || len(fixture.Cases) != 12 {
		t.Fatal("unexpected fixture provenance")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			got, err := DecodeTextStateRequest(strings.NewReader(string(tc.Request)))
			if !tc.Accepted {
				if err == nil || len(got.Questions) != 0 {
					t.Fatalf("accepted rejected case: %+v", got)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got.Model != tc.Model || got.State != tc.State || got.RawLogits != tc.RawLogits || len(got.Questions) != len(tc.Questions) {
				t.Fatalf("header/count mismatch: %+v", got)
			}
			for i, want := range tc.Questions {
				q := got.Questions[i]
				if q.ID != want.ID || q.Type != want.Type || !sameOptionalText(q.Instructions, want.Instructions) {
					t.Fatalf("question %d differs: %+v", i, q)
				}
				switch q.Type {
				case "choice":
					if len(q.Choices) != 2 || q.Choices[0].ID != "first" || q.Choices[0].Description != nil || q.Choices[1].ID != "second" || q.Choices[1].Description == nil || *q.Choices[1].Description != "text" {
						t.Fatalf("choice order/null changed: %+v", q)
					}
				case "score":
					if len(q.Levels) != 3 || q.Levels[0] == nil || *q.Levels[0] != "low" || q.Levels[1] != nil || q.Levels[2] == nil || *q.Levels[2] != "high" {
						t.Fatalf("score order/null changed: %+v", q)
					}
				case "noul":
					if len(q.NoulCriteria) != 1 || q.NoulCriteria[0].ID != "false" || q.NoulCriteria[0].Description != nil {
						t.Fatalf("Noul criteria changed: %+v", q)
					}
				}
			}
		})
	}
}

func sameOptionalText(a, b *string) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return *a == *b
}

func TestTextStateRejectsUnsupportedAndMalformed(t *testing.T) {
	base := `{"model":"fixture","state":"","questions":{"pick":{"type":"choice","instructions":"","criteria":{"one":null,"two":"two"}}}}`
	for name, body := range map[string]string{
		"nil":                    "",
		"duplicate top":          `{"model":"fixture","model":"other","state":"","questions":{}}`,
		"duplicate question":     `{"model":"fixture","state":"","questions":{"a":{"type":"noul","instructions":""},"a":{"type":"noul","instructions":""}}}`,
		"duplicate choice":       `{"model":"fixture","state":"","questions":{"a":{"type":"choice","instructions":"","criteria":{"a":null,"a":null}}}}`,
		"trailing":               base + ` {}`,
		"messages":               strings.Replace(base, `"state":""`, `"messages":[{"role":"user","content":"hi"}]`, 1),
		"structured state":       strings.Replace(base, `"state":""`, `"state":{"x":1}`, 1),
		"structured instruction": strings.Replace(base, `"instructions":""`, `"instructions":{"x":1}`, 1),
		"null state":             strings.Replace(base, `"state":""`, `"state":null`, 1),
		"missing model":          strings.Replace(base, `"model":"fixture",`, "", 1),
		"invalid UTF8":           base + "\xff",
		"options extra":          strings.Replace(base, `"questions":`, `"options":{"unknown":true},"questions":`, 1),
		"options null":           strings.Replace(base, `"questions":`, `"options":{"raw_logits":null},"questions":`, 1),
		"options wrong type":     strings.Replace(base, `"questions":`, `"options":{"raw_logits":1},"questions":`, 1),
		"Noul extra":             `{"model":"fixture","state":"","questions":{"x":{"type":"noul","instructions":"","criteria":{"perhaps":null}}}}`,
		"null choice":            strings.Replace(base, `"criteria":{"one":null,"two":"two"}`, `"criteria":null`, 1),
		"null model":             strings.Replace(base, `"model":"fixture"`, `"model":null`, 1),
		"null questions":         strings.Replace(base, `"questions":{"pick":`, `"questions":null,"other":{"pick":`, 1),
		"wrong score type":       strings.Replace(base, `"type":"choice"`, `"type":"score"`, 1),
	} {
		t.Run(name, func(t *testing.T) {
			if name == "nil" {
				if _, err := DecodeTextStateRequest(nil); err == nil {
					t.Fatal("accepted nil")
				}
				return
			}
			if got, err := DecodeTextStateRequest(strings.NewReader(body)); err == nil {
				t.Fatalf("accepted malformed request %+v", got)
			}
		})
	}
	if _, err := DecodeTextStateRequest(strings.NewReader(strings.Repeat(" ", MaxRequestBytes+1))); err == nil {
		t.Fatal("accepted oversized request")
	}
	for _, n := range []int{2, 50} {
		criteria := make([]string, n)
		for i := range criteria {
			criteria[i] = fmt.Sprintf("%q", fmt.Sprint(i))
		}
		body := fmt.Sprintf(`{"model":"fixture","state":"","questions":{"score":{"type":"score","instructions":null,"criteria":[%s]}}}`, strings.Join(criteria, ","))
		got, err := DecodeTextStateRequest(strings.NewReader(body))
		if err != nil || len(got.Questions) != 1 || len(got.Questions[0].Levels) != n {
			t.Fatalf("score %d: %+v %v", n, got, err)
		}
	}
}
