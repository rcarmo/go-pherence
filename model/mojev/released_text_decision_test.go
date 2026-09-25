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

func TestReleasedMoJevTextDecision(t *testing.T) {
	for _, pin := range []struct{ path, hash string }{{"../../scripts/mojev_oracle_released_text_decision.py", "9e3be12badfb265ddd696b192a01259c44b9d42b110089eef6e4f4f055be8fad"}, {"testdata/released_text_decision.json", "f0144ef90ef5443c001bce5bdc0259bdfd81457a5266654e4a13e7eb0abbd6f9"}} {
		data, err := os.ReadFile(pin.path)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(data)) != pin.hash {
			t.Fatalf("oracle hash changed: %s", pin.path)
		}
	}
	data, err := os.ReadFile("testdata/released_text_decision.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema         int             `json:"schema"`
		Revision       string          `json:"source_revision"`
		ModelSHA       string          `json:"modeling_sha256"`
		ServeSHA       string          `json:"serve_sha256"`
		FullSHA        string          `json:"full_sha256"`
		ConfigSHA      string          `json:"config_sha256"`
		WeightsSHA     string          `json:"weights_sha256"`
		WeightsSize    int64           `json:"weights_size"`
		TokenizerSHA   string          `json:"tokenizer_sha256"`
		TokenConfigSHA string          `json:"tokenizer_config_sha256"`
		Request        json.RawMessage `json:"request"`
		IDs            []int           `json:"packed_ids"`
		Logits         [][]float64     `json:"sorted_logits"`
		Response       struct {
			Model   string                    `json:"model"`
			Answers map[string]map[string]any `json:"answers"`
			Usage   struct {
				Input  int `json:"input_tokens"`
				Output int `json:"output_tokens"`
			} `json:"usage"`
		} `json:"response"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != 1 || fixture.Revision != SourceRevision || fixture.ModelSHA != "a8e93f62d92c6748c5d001fef4f9516d6a74b10158d7265f53bab13f1091d458" || fixture.ServeSHA != "7d244785d11eb5cdb490080f24f1307b2c5df8126a6b0858c4f1db341d838c8d" || fixture.FullSHA != "5e2972605c190a511cdfb3a1c01358c4855b229c93a1679a6c569c5b43b07b79" || fixture.ConfigSHA != "1b3fb0dd8ae5a1e334b31bae1cfb2e8212231bd549884819304f29334313f4c9" || fixture.WeightsSHA != "eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50" || fixture.WeightsSize != 1710234304 || fixture.TokenizerSHA != "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523" || fixture.TokenConfigSHA != "66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b" || len(fixture.Logits) != 3 || len(fixture.IDs) != 60 {
		t.Fatal("unexpected released fixture provenance")
	}
	path := os.Getenv("GO_PHERENCE_MOJEV_CHECKPOINT_DIR")
	if path == "" {
		t.Skip("set GO_PHERENCE_MOJEV_CHECKPOINT_DIR to approved checkpoint and tokenizer")
	}
	for _, asset := range []struct {
		name, hash string
		size       int64
	}{{"model.safetensors", fixture.WeightsSHA, fixture.WeightsSize}, {"tokenizer.json", fixture.TokenizerSHA, 19989325}, {"tokenizer_config.json", fixture.TokenConfigSHA, 1124}} {
		file, err := os.Open(filepath.Join(path, asset.name))
		if err != nil {
			t.Fatal(err)
		}
		info, err := file.Stat()
		if err != nil || info.Size() != asset.size {
			file.Close()
			t.Fatalf("%s size mismatch: %v", asset.name, err)
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
	fields := make([]TextField, len(req.Fields))
	menus := make([][]string, len(req.Fields))
	for i, f := range req.Fields {
		fields[i] = TextField{Name: f.ID, Description: f.Instructions, Options: f.SortedOptions}
		menus[i] = f.SortedOptions
	}
	encode := func(s string) ([]int, error) { return tok.Encode(s), nil }
	packed, err := PackTextRows([]TextRow{{State: req.State, Menus: menus}}, fields, 128, 32, 248044, encode)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(packed.IDs[0], fixture.IDs) {
		t.Fatalf("released packed IDs differ: got %v want %v", packed.IDs[0], fixture.IDs)
	}
	got, err := AssembleTextDecision(req, fixture.Logits, encode, 248044, 128, 32)
	if err != nil {
		t.Fatal(err)
	}
	if got.Model != fixture.Response.Model || got.Usage.InputTokens != fixture.Response.Usage.Input || got.Usage.OutputTokens != fixture.Response.Usage.Output || len(got.Answers) != len(fixture.Response.Answers) {
		t.Fatalf("response shape/usage: %+v", got)
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
