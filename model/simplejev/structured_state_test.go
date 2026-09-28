package simplejev

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"strings"
	"sync"
	"testing"
)

func TestStructuredStateUpstreamFixture(t *testing.T) {
	const scriptHash = "64a4f9571a77a7d73c54728d5a81617bdd97971f585ef5830f29df2a4c064bd6"
	const fixtureHash = "36a6b30e9587146ad11d6ed179270a7992f7519e834a4018e6b2377255f356f0"
	for _, item := range []struct{ file, hash string }{
		{"../../scripts/simplejev_oracle_structured_state.py", scriptHash},
		{"testdata/upstream_structured_state_v1.json", fixtureHash},
	} {
		data, err := os.ReadFile(item.file)
		if err != nil {
			t.Fatal(err)
		}
		actual := sha256.Sum256(data)
		if hex.EncodeToString(actual[:]) != item.hash {
			t.Fatalf("%s hash changed", item.file)
		}
	}
	data, err := os.ReadFile("testdata/upstream_structured_state_v1.json")
	if err != nil {
		t.Fatal(err)
	}
	// Use explicit field names so fixture changes cannot silently drop cases.
	var cases struct {
		Schema           int    `json:"schema"`
		Revision         string `json:"revision"`
		RequestSchemaSHA string `json:"request_schema_sha256"`
		PromptBuilderSHA string `json:"prompt_builder_sha256"`
		Accepted         []struct {
			Input     string `json:"input"`
			Canonical string `json:"canonical"`
			Kind      string `json:"kind"`
		} `json:"accepted"`
		Rejected []struct {
			Name      string `json:"name"`
			Input     string `json:"input"`
			ErrorType string `json:"error_type"`
		} `json:"rejected"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if cases.Schema != 1 || cases.Revision != "b02aa81c915a8193759b3cd33fef74721d6e005b" ||
		cases.RequestSchemaSHA != "6fa1c1215e8fc9de7aedbee77db6520bbb811922df9c45858bf78c4e4a02ceac" ||
		cases.PromptBuilderSHA != "14a885d3bfa44b0c8aa0ffc9ce84611a459957d938e1847f96d93c7f1840fa3d" ||
		len(cases.Accepted) != 6 || len(cases.Rejected) != 4 {
		t.Fatalf("unexpected structured-state provenance or case count: %+v", cases)
	}
	for _, c := range cases.Accepted {
		t.Run(c.Kind+"/"+c.Input, func(t *testing.T) {
			got, err := CanonicalStructuredState(strings.NewReader(c.Input))
			if err != nil || got != c.Canonical {
				t.Fatalf("canonical: got %q, %v; want %q", got, err, c.Canonical)
			}
		})
	}
	for _, c := range cases.Rejected {
		t.Run(c.Name, func(t *testing.T) {
			if c.ErrorType != "ValidationError" {
				t.Fatalf("unexpected upstream error type %q", c.ErrorType)
			}
			if got, err := CanonicalStructuredState(strings.NewReader(c.Input)); err == nil || got != "" {
				t.Fatalf("accepted rejected state: %q, %v", got, err)
			}
		})
	}
}

func TestStructuredStateBoundaries(t *testing.T) {
	valid := []struct{ input, output string }{
		{`{"z":-0,"a":[9223372036854775807,-9223372036854775808]}`, `{"a":[9223372036854775807,-9223372036854775808],"z":0}`},
		{`{"quoted":"<>&/é","escape":"\b\f\n\r\t"}`, `{"escape":"\b\f\n\r\t","quoted":"<>&/é"}`},
		{`""`, `""`},
		{strings.Repeat("[", MaxStructuredStateDepth) + "0" + strings.Repeat("]", MaxStructuredStateDepth), strings.Repeat("[", MaxStructuredStateDepth) + "0" + strings.Repeat("]", MaxStructuredStateDepth)},
		{`{}` + strings.Repeat(" ", MaxStructuredStateBytes-2), `{}`},
	}
	for _, c := range valid {
		got, err := CanonicalStructuredState(strings.NewReader(c.input))
		if err != nil || got != c.output {
			t.Errorf("%q -> %q, %v; want %q", c.input[:min(len(c.input), 80)], got, err, c.output)
		}
	}
	invalid := []string{
		``, `null`, `true`, `42`, `1.5`, `{"n":1.0}`, `{"n":1e2}`, `{"n":NaN}`,
		`{"n":9223372036854775808}`, `{"n":-9223372036854775809}`,
		`{"x":1,"x":2}`, `{"x":{"a":0,"a":1}}`,
		`{"x":{}`, `[] []`, `{} null`, `{"x":true} trailing`, `{"x":`,
		`{"x":"\ud800"}`, `"\ufffd"`, `"\u2028"`, `"\u2029"`,
		`{"a":0,"\u2029":1}`, `"` + string([]byte{0xff}) + `"`,
		strings.Repeat("[", MaxStructuredStateDepth+1) + "0" + strings.Repeat("]", MaxStructuredStateDepth+1),
		`"` + strings.Repeat("a", MaxStructuredStateBytes) + `"`,
	}
	for _, input := range invalid {
		if got, err := CanonicalStructuredState(strings.NewReader(input)); err == nil || got != "" {
			t.Errorf("accepted %q: %q, %v", input[:min(len(input), 80)], got, err)
		}
	}
	if got, err := CanonicalStructuredState(nil); err == nil || got != "" {
		t.Fatalf("nil reader: %q, %v", got, err)
	}
	if got, err := CanonicalStructuredState(failingStructuredReader{}); err == nil || got != "" {
		t.Fatalf("failing reader: %q, %v", got, err)
	}
}

type failingStructuredReader struct{}

func (failingStructuredReader) Read([]byte) (int, error) { return 0, errors.New("read failed") }

func TestStructuredStateOwnedAndConcurrent(t *testing.T) {
	input := []byte(`{"z":4,"a":"café"}`)
	got, err := CanonicalStructuredState(bytes.NewReader(input))
	if err != nil {
		t.Fatal(err)
	}
	for i := range input {
		input[i] = 'x'
	}
	if got != `{"a":"café","z":4}` {
		t.Fatal("output changed after input mutation")
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 30 {
				actual, err := CanonicalStructuredState(strings.NewReader(`{"z":4,"a":"café"}`))
				if err != nil || actual != got {
					t.Errorf("concurrent canonical: %q, %v", actual, err)
				}
			}
		}()
	}
	wg.Wait()
}

var _ io.Reader = failingStructuredReader{}
