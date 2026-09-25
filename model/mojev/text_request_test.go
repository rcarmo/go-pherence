package mojev

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestPinnedMoJevTextRequests(t *testing.T) {
	for _, pin := range []struct{ path, hash string }{{"../../scripts/mojev_oracle_text_request.py", "d9871a4299fa91e970e86608b916dc81a52e23835a41dac83948d93f9ec1e080"}, {"testdata/text_request.json", "d5d75552cd485a1764ceb8f823d5e108db86112d4bf058c1c1808077b01eba05"}} {
		data, err := os.ReadFile(pin.path)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(data)) != pin.hash {
			t.Fatalf("oracle hash changed: %s", pin.path)
		}
	}
	data, err := os.ReadFile("testdata/text_request.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema   int    `json:"schema"`
		Revision string `json:"source_revision"`
		ServeSHA string `json:"serve_sha256"`
		FullSHA  string `json:"full_sha256"`
		Cases    []struct {
			Name    string          `json:"name"`
			Request json.RawMessage `json:"request"`
			State   string          `json:"state_text"`
			Fields  []struct {
				ID           string   `json:"id"`
				Kind         string   `json:"kind"`
				Keys         []string `json:"keys"`
				Options      []string `json:"options"`
				Order        []int    `json:"sorted_indices"`
				Sorted       []string `json:"sorted_options"`
				Instructions string   `json:"instructions"`
			} `json:"fields"`
		} `json:"cases"`
		Invalid []struct {
			Name     string          `json:"name"`
			Question json.RawMessage `json:"question"`
		} `json:"invalid"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != 1 || fixture.Revision != SourceRevision || fixture.ServeSHA != "7d244785d11eb5cdb490080f24f1307b2c5df8126a6b0858c4f1db341d838c8d" || fixture.FullSHA != "5e2972605c190a511cdfb3a1c01358c4855b229c93a1679a6c569c5b43b07b79" || len(fixture.Cases) != 3 || len(fixture.Invalid) != 5 {
		t.Fatal("unexpected request oracle provenance")
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			got, err := DecodeTextRequest(bytes.NewReader(tc.Request))
			if err != nil {
				t.Fatal(err)
			}
			if got.Model != "mojev-latest" || got.State != tc.State || len(got.Fields) != len(tc.Fields) {
				t.Fatalf("request mismatch: %+v", got)
			}
			for i, want := range tc.Fields {
				f := got.Fields[i]
				if f.ID != want.ID || f.Kind != want.Kind || f.Instructions != want.Instructions || !reflect.DeepEqual(f.Keys, want.Keys) || !reflect.DeepEqual(f.Options, want.Options) || !reflect.DeepEqual(f.SortedIndices, want.Order) || !reflect.DeepEqual(f.SortedOptions, want.Sorted) {
					t.Fatalf("field[%d] mismatch: %+v want %+v", i, f, want)
				}
			}
			original := got.Fields[0].Options[0]
			for i := range tc.Request {
				tc.Request[i] = ' '
			}
			if got.Fields[0].Options[0] != original {
				t.Fatal("returned result aliases input bytes")
			}
		})
	}
	for _, tc := range fixture.Invalid {
		t.Run("upstream_rejects/"+tc.Name, func(t *testing.T) {
			body := append([]byte(`{"model":"mojev-latest","state":"s","questions":{"q":`), tc.Question...)
			body = append(body, []byte("}}")...)
			got, err := DecodeTextRequest(bytes.NewReader(body))
			if err == nil || !reflect.DeepEqual(got, TextRequest{}) {
				t.Fatalf("accepted upstream invalid question: %+v %v", got, err)
			}
		})
	}
}

func TestDecodeTextRequestRejectsMalformed(t *testing.T) {
	for name, raw := range map[string]string{
		"nil body": "", "nonobject": "[]", "missing model": `{"state":"s","questions":{"q":{"type":"noul"}}}`,
		"missing state":            `{"model":"m","questions":{"q":{"type":"noul"}}}`,
		"structured state":         `{"model":"m","state":{"text":"s"},"questions":{"q":{"type":"noul"}}}`,
		"empty questions":          `{"model":"m","state":"s","questions":{}}`,
		"duplicate model":          `{"model":"m","model":"n","state":"s","questions":{"q":{"type":"noul"}}}`,
		"duplicate question":       `{"model":"m","state":"s","questions":{"q":{"type":"noul"},"q":{"type":"noul"}}}`,
		"trailing":                 `{"model":"m","state":"s","questions":{"q":{"type":"noul"}}} {}`,
		"unknown top":              `{"model":"m","state":"s","messages":[],"questions":{"q":{"type":"noul"}}}`,
		"unknown question":         `{"model":"m","state":"s","questions":{"q":{"type":"noul","media":{}}}}`,
		"bad kind":                 `{"model":"m","state":"s","questions":{"q":{"type":"other"}}}`,
		"choice empty":             `{"model":"m","state":"s","questions":{"q":{"type":"choice","criteria":{}}}}`,
		"score empty":              `{"model":"m","state":"s","questions":{"q":{"type":"score","criteria":[]}}}`,
		"noul wrong criteria":      `{"model":"m","state":"s","questions":{"q":{"type":"noul","criteria":["x"]}}}`,
		"bad UTF-8":                string([]byte{0xff}),
		"image marker":             `{"model":"m","state":"<|image_pad|>","questions":{"q":{"type":"noul"}}}`,
		"duplicate candidate text": `{"model":"m","state":"s","questions":{"q":{"type":"choice","criteria":{"a":"same","a: same":null}}}}`,
		"structured score level":   `{"model":"m","state":"s","questions":{"q":{"type":"score","criteria":["low",{"high":true}]}}}`,
		"too large":                strings.Repeat(" ", maxTextRequestBytes+1),
	} {
		t.Run(name, func(t *testing.T) {
			got, err := DecodeTextRequest(strings.NewReader(raw))
			if err == nil || !reflect.DeepEqual(got, TextRequest{}) {
				t.Fatalf("accepted invalid request: %+v %v", got, err)
			}
		})
	}
	if got, err := DecodeTextRequest(nil); err == nil || !reflect.DeepEqual(got, TextRequest{}) {
		t.Fatal("accepted nil reader")
	}
}
