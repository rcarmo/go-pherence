package mojev

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestPinnedMoJevTextDecision(t *testing.T) {
	for _, pin := range []struct{ path, hash string }{
		{"../../scripts/mojev_oracle_text_response.py", "8bbb2a6ede0b0119313cdbd9f9b850dc28f17940198c35d96b44b12bcbcf0708"},
		{"testdata/text_response.json", "335b4055bc90a32f5c9bdbc4001c1adb7d9dc1fa1283404830de80a32a5043c5"},
	} {
		data, err := os.ReadFile(pin.path)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(data)) != pin.hash {
			t.Fatalf("oracle hash changed: %s", pin.path)
		}
	}
	data, err := os.ReadFile("testdata/text_response.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema       int             `json:"schema"`
		Revision     string          `json:"source_revision"`
		ServeSHA     string          `json:"serve_sha256"`
		FullSHA      string          `json:"full_sha256"`
		TokenizerSHA string          `json:"tokenizer_sha256"`
		ConfigSHA    string          `json:"tokenizer_config_sha256"`
		Request      json.RawMessage `json:"request"`
		Logits       [][]float64     `json:"sorted_logits"`
		InputTokens  int             `json:"input_tokens"`
		Response     struct {
			Model   string                    `json:"model"`
			Answers map[string]map[string]any `json:"answers"`
			Usage   struct {
				InputTokens  int `json:"input_tokens"`
				OutputTokens int `json:"output_tokens"`
			} `json:"usage"`
		} `json:"response"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != 1 || fixture.Revision != SourceRevision || fixture.ServeSHA != "7d244785d11eb5cdb490080f24f1307b2c5df8126a6b0858c4f1db341d838c8d" || fixture.FullSHA != "5e2972605c190a511cdfb3a1c01358c4855b229c93a1679a6c569c5b43b07b79" || fixture.TokenizerSHA != "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523" || fixture.ConfigSHA != "66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b" || len(fixture.Logits) != 3 || fixture.InputTokens != 60 {
		t.Fatal("unexpected oracle provenance")
	}
	path := os.Getenv("GO_PHERENCE_MOJEV_CHECKPOINT_DIR")
	if path == "" {
		t.Skip("set GO_PHERENCE_MOJEV_CHECKPOINT_DIR to approved tokenizer assets")
	}
	for _, asset := range []struct{ name, hash string }{{"tokenizer.json", fixture.TokenizerSHA}, {"tokenizer_config.json", fixture.ConfigSHA}} {
		file, err := os.Open(filepath.Join(path, asset.name))
		if err != nil {
			t.Fatal(err)
		}
		digest := sha256.New()
		_, err = io.Copy(digest, file)
		closeErr := file.Close()
		if err != nil {
			t.Fatal(err)
		}
		if closeErr != nil {
			t.Fatal(closeErr)
		}
		if fmt.Sprintf("%x", digest.Sum(nil)) != asset.hash {
			t.Fatalf("%s hash mismatch", asset.name)
		}
	}
	tok, err := tokenizer.LoadWithConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	req, err := DecodeTextRequest(bytes.NewReader(fixture.Request))
	if err != nil {
		t.Fatal(err)
	}
	before := make([][]float64, len(fixture.Logits))
	for i, row := range fixture.Logits {
		before[i] = append([]float64(nil), row...)
	}
	got, err := AssembleTextDecision(req, fixture.Logits, func(s string) ([]int, error) { return tok.Encode(s), nil }, 248044, 128, 32)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(fixture.Logits, before) || got.Model != fixture.Response.Model || got.Usage.InputTokens != fixture.Response.Usage.InputTokens || got.Usage.OutputTokens != 0 || len(got.Answers) != len(fixture.Response.Answers) {
		t.Fatalf("decision shape/usage mismatch: %+v", got)
	}
	for id, want := range fixture.Response.Answers {
		answer, ok := got.Answers[id]
		if !ok || len(answer) != len(want) {
			t.Fatalf("answer %s shape mismatch: %v", id, answer)
		}
		for name, value := range want {
			actual, ok := answer[name]
			if !ok {
				t.Fatalf("missing %s.%s", id, name)
			}
			switch expected := value.(type) {
			case string:
				if actual != expected {
					t.Fatalf("%s.%s=%v want %s", id, name, actual, expected)
				}
			case float64:
				if math.Abs(actual.(float64)-expected) > 1e-6 {
					t.Fatalf("%s.%s=%v want %g", id, name, actual, expected)
				}
			case map[string]any:
				switch m := actual.(type) {
				case map[string]float64:
					if len(m) != len(expected) {
						t.Fatal("probability count")
					}
					for k, v := range expected {
						if math.Abs(m[k]-v.(float64)) > 1e-6 {
							t.Fatalf("%s.%s[%s]=%v want %v", id, name, k, m[k], v)
						}
					}
				case map[string]string:
					if len(m) != len(expected) {
						t.Fatal("legend count")
					}
					for k, v := range expected {
						if m[k] != v {
							t.Fatalf("%s.%s[%s]=%v want %v", id, name, k, m[k], v)
						}
					}
				default:
					t.Fatalf("unexpected %s.%s type %T", id, name, actual)
				}
			}
		}
	}
}

func TestAssembleTextDecisionRejectsMalformed(t *testing.T) {
	req, err := DecodeTextRequest(bytes.NewBufferString(`{"model":"m","state":"s","questions":{"a":{"type":"choice","criteria":{"x":"first","y":"second"}},"b":{"type":"noul"}}}`))
	if err != nil {
		t.Fatal(err)
	}
	enc := func(s string) ([]int, error) { return []int{len(s)}, nil }
	valid := [][]float64{{1, 2}, {3, 4}}
	for name, tc := range map[string]struct {
		req    TextRequest
		rows   [][]float64
		encode TextEncoder
	}{
		"missing rows":         {req, valid[:1], enc},
		"nil encoder":          {req, valid, nil},
		"nonfinite second row": {req, [][]float64{{1, 2}, {math.NaN(), 4}}, enc},
		"short second row":     {req, [][]float64{{1, 2}, {3}}, enc},
	} {
		t.Run(name, func(t *testing.T) {
			got, err := AssembleTextDecision(tc.req, tc.rows, tc.encode, 0, 128, 32)
			if err == nil || got != nil {
				t.Fatalf("accepted malformed decision: %+v", got)
			}
		})
	}
	bad := req
	bad.Fields = append([]TextRequestField(nil), req.Fields...)
	bad.Fields[0].SortedIndices = []int{0, 0}
	if got, err := AssembleTextDecision(bad, valid, enc, 0, 128, 32); err == nil || got != nil {
		t.Fatal("accepted duplicate sorted index")
	}
	bad.Fields[0].SortedIndices = append([]int(nil), req.Fields[0].SortedIndices...)
	bad.Fields[0].SortedOptions = append([]string(nil), req.Fields[0].SortedOptions...)
	bad.Fields[0].SortedIndices[0], bad.Fields[0].SortedIndices[1] = bad.Fields[0].SortedIndices[1], bad.Fields[0].SortedIndices[0]
	bad.Fields[0].SortedOptions[0], bad.Fields[0].SortedOptions[1] = bad.Fields[0].SortedOptions[1], bad.Fields[0].SortedOptions[0]
	if got, err := AssembleTextDecision(bad, valid, enc, 0, 128, 32); err == nil || got != nil {
		t.Fatal("accepted noncanonical candidate order")
	}
}
