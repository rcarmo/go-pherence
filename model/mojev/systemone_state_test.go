package mojev

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func stateRequest(state string) string {
	return `{"model":"m","state":` + state + `,"questions":{"q":{"type":"noul"}}}`
}

func TestSystemOneStateOracle(t *testing.T) {
	for path, want := range map[string]string{"testdata/gso_state_render.json": "11c2dbe0a1a4038aab68c0059ec6ab826dbbb529afbe07a0c101ed46ebc30f43", "../../scripts/mojev_gso_state_oracle.ts": "64760b6be20f9e149b484950f200757b0a0f52a9500004a2353a3bb838498897"} {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(b)) != want {
			t.Fatal("fixture hash", path)
		}
	}
	var f struct {
		Schema   int    `json:"schema"`
		Revision string `json:"source_revision"`
		SHA      string `json:"source_sha256"`
		Cases    []struct {
			Name, Raw, Text string
			Valid           bool
		}
	}
	b, err := os.ReadFile("testdata/gso_state_render.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	if f.Schema != 1 || f.Revision != "b18ee0d4748bac436999aa72c000e06406c3cce6" || f.SHA != "1cd60986f1a3e37d07f7743e9e126d26a96f54047527836062b453e410b7fde8" || len(f.Cases) != 10 {
		t.Fatal("provenance")
	}
	for _, c := range f.Cases {
		t.Run(c.Name, func(t *testing.T) {
			req, err := DecodeSystemOneTextRequest(strings.NewReader(stateRequest(c.Raw)))
			if c.Valid {
				if err != nil || req.State != c.Text {
					t.Fatalf("got %q %v want %q", req.State, err, c.Text)
				}
			} else if err == nil || !reflect.DeepEqual(req, TextRequest{}) {
				t.Fatal("invalid returned partial", req, err)
			}
		})
	}
	// Existing plain-text decoder and request semantics are unchanged.
	text := stateRequest(`"ordinary {not JSON}"`)
	a, err := DecodeTextRequest(strings.NewReader(text))
	if err != nil {
		t.Fatal(err)
	}
	bReq, err := DecodeSystemOneTextRequest(strings.NewReader(text))
	if err != nil || !reflect.DeepEqual(a, bReq) {
		t.Fatal("text behavior changed")
	}
	// Text input keeps the old decoded-size limit even with long escape syntax.
	escaped := stateRequest(`"` + strings.Repeat(`\u0061`, 12000) + `"`)
	old, oldErr := DecodeTextRequest(strings.NewReader(escaped))
	newReq, newErr := DecodeSystemOneTextRequest(strings.NewReader(escaped))
	if oldErr != nil || newErr != nil || !reflect.DeepEqual(old, newReq) {
		t.Fatal("escaped plain-text compatibility", oldErr, newErr)
	}
	if _, err := DecodeTextRequest(strings.NewReader(stateRequest(`{"x":1}`))); err == nil {
		t.Fatal("strict decoder broadened")
	}
}

func TestSystemOneStateRejects(t *testing.T) {
	for _, raw := range []string{`{"a":1,"a":2}`, `{"a":{"x":1,"\u0078":2}}`, `[true`, `{"x":}`, `false`, `1`, strings.Repeat("[", 33) + "0" + strings.Repeat("]", 33), `{"x":"` + strings.Repeat("a", 64<<10) + `"}`} {
		req, err := DecodeSystemOneTextRequest(strings.NewReader(stateRequest(raw)))
		if err == nil || !reflect.DeepEqual(req, TextRequest{}) {
			t.Fatal("invalid accepted", len(raw), err)
		}
	}
	raw := strings.Repeat("[", 32) + "0" + strings.Repeat("]", 32)
	if _, err := DecodeSystemOneTextRequest(strings.NewReader(stateRequest(raw))); err != nil {
		t.Fatal("depth32", err)
	}
	if _, err := DecodeSystemOneTextRequest(nil); err == nil {
		t.Fatal("nil reader")
	}
	// Validate internal walker malformed-token/error branches without exposing it.
	for _, raw := range []string{``, `}`, `{"x"`, `{"x":`, `{"x":1`, `[1,`, `{} []`} {
		if err := validateStructuredState([]byte(raw), nil); err == nil {
			t.Fatal("invalid internal state", raw)
		}
	}
}

func TestSystemOneStateControlsAndOwnership(t *testing.T) {
	tok := &tokenizer.Tokenizer{AddedSpecial: map[string]int{"<|reserved|>": 42}}
	for _, raw := range []string{`{"x":"\u003c|reserved|>"}`, `{"\u003c|reserved|>":0}`, `[{"x":"<|reserved|>"}]`} {
		req, err := DecodeSystemOneTextRequest(strings.NewReader(stateRequest(raw)))
		if err != nil {
			t.Fatal(err)
		}
		if err = ValidateTextControls(req, tok); err == nil {
			t.Fatal("escaped reserved marker accepted")
		}
	}
	var visited []string
	if err := validateStructuredState([]byte(`{"a":["b",null,true,1]}`), func(s string) error { visited = append(visited, s); return nil }); err != nil || !reflect.DeepEqual(visited, []string{"a", "b"}) {
		t.Fatal("walk", visited, err)
	}
	sentinel := errors.New("visit")
	if err := validateStructuredState([]byte(`["x"]`), func(string) error { return sentinel }); err != sentinel {
		t.Fatal("visitor error")
	}
	req, err := DecodeSystemOneTextRequest(strings.NewReader(stateRequest(`{"x":[1,"café"],"n":null}`)))
	if err != nil {
		t.Fatal(err)
	}
	if err = ValidateTextControls(req, tok); err != nil {
		t.Fatal(err)
	}
	req.State = `{"x":"\u003c|reserved|>"}`
	if err = ValidateTextControls(req, tok); err == nil {
		t.Fatal("caller mutation bypassed checks")
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 10 {
				r, e := DecodeSystemOneTextRequest(strings.NewReader(stateRequest(`{"x":1}`)))
				if e != nil || r.State != `{"x":1}` {
					t.Error("concurrent decode", e)
				}
			}
		}()
	}
	wg.Wait()
}
