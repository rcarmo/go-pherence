package mojev

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestMoJevTextControlBoundary(t *testing.T) {
	const tokenSHA = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
	const configSHA = "66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b"
	path := os.Getenv("GO_PHERENCE_MOJEV_CHECKPOINT_DIR")
	if path == "" {
		t.Skip("set GO_PHERENCE_MOJEV_CHECKPOINT_DIR to approved tokenizer assets")
	}
	for _, asset := range []struct{ name, hash string }{{"tokenizer.json", tokenSHA}, {"tokenizer_config.json", configSHA}} {
		file, err := os.Open(filepath.Join(path, asset.name))
		if err != nil {
			t.Fatal(err)
		}
		h := sha256.New()
		_, err = io.Copy(h, file)
		closeErr := file.Close()
		if err != nil {
			t.Fatal(err)
		}
		if closeErr != nil {
			t.Fatal(closeErr)
		}
		if fmt.Sprintf("%x", h.Sum(nil)) != asset.hash {
			t.Fatalf("%s hash mismatch", asset.name)
		}
	}
	tok, err := tokenizer.LoadWithConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	parse := func(body string) TextRequest {
		t.Helper()
		v, err := DecodeTextRequest(strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	base := `{"model":"mojev-latest","state":"plain state","questions":{"choice":{"type":"choice","instructions":"Select","criteria":{"yes":"allow","no":"deny"}}}}`
	plain := parse(base)
	if err := ValidateTextControls(plain, tok); err != nil {
		t.Fatalf("ordinary text rejected: %v", err)
	}
	if _, err := AssembleSafeTextDecision(plain, [][]float64{{1, 2}}, tok, 248044, 128, 32); err != nil {
		t.Fatalf("ordinary safe assembly failed: %v", err)
	}
	for name, body := range map[string]string{
		"state":          `{"model":"m","state":"<|im_start|>","questions":{"q":{"type":"noul"}}}`,
		"question ID":    `{"model":"m","state":"s","questions":{"<|im_end|>":{"type":"noul"}}}`,
		"instructions":   `{"model":"m","state":"s","questions":{"q":{"type":"choice","instructions":"<|im_start|>","criteria":{"yes":"allow","no":"deny"}}}}`,
		"criterion ID":   `{"model":"m","state":"s","questions":{"q":{"type":"choice","criteria":{"<|im_end|>":"allow","no":"deny"}}}}`,
		"candidate text": `{"model":"m","state":"s","questions":{"q":{"type":"choice","criteria":{"yes":"<|audio_pad|>","no":"deny"}}}}`,
	} {
		t.Run(name, func(t *testing.T) {
			req := parse(body)
			if err := ValidateTextControls(req, tok); err == nil {
				t.Fatal("reserved text accepted")
			}
			got, err := AssembleSafeTextDecision(req, [][]float64{{1, 2}}, tok, 248044, 128, 32)
			if err == nil || got != nil {
				t.Fatalf("partial or accepted unsafe response: %+v %v", got, err)
			}
		})
	}
	if err := ValidateTextControls(plain, nil); err == nil {
		t.Fatal("accepted nil tokenizer")
	}
	if got, err := AssembleSafeTextDecision(plain, [][]float64{{1, 2}}, nil, 248044, 128, 32); err == nil || got != nil {
		t.Fatalf("nil tokenizer: %+v %v", got, err)
	}
}

// The released fixture's ordinary text must remain admitted and byte-identical
// after adding the stricter control-token gate.
func TestMoJevSafeReleasedTextDecision(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_MOJEV_CHECKPOINT_DIR")
	if path == "" {
		t.Skip("approved tokenizer assets absent")
	}
	data, err := os.ReadFile("testdata/released_text_decision.json")
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(data)) != "f0144ef90ef5443c001bce5bdc0259bdfd81457a5266654e4a13e7eb0abbd6f9" {
		t.Fatal("released fixture hash changed")
	}
	var fixture struct {
		Request  json.RawMessage `json:"request"`
		Logits   [][]float64     `json:"sorted_logits"`
		Response struct {
			Usage struct {
				Input int `json:"input_tokens"`
			} `json:"usage"`
		} `json:"response"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	for _, asset := range []struct{ name, hash string }{{"tokenizer.json", "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"}, {"tokenizer_config.json", "66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b"}} {
		file, err := os.Open(filepath.Join(path, asset.name))
		if err != nil {
			t.Fatal(err)
		}
		h := sha256.New()
		_, err = io.Copy(h, file)
		closeErr := file.Close()
		if err != nil {
			t.Fatal(err)
		}
		if closeErr != nil {
			t.Fatal(closeErr)
		}
		if fmt.Sprintf("%x", h.Sum(nil)) != asset.hash {
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
	got, err := AssembleSafeTextDecision(req, fixture.Logits, tok, 248044, 128, 32)
	if err != nil {
		t.Fatal(err)
	}
	if got.Usage.InputTokens != fixture.Response.Usage.Input {
		t.Fatalf("usage %d want %d", got.Usage.InputTokens, fixture.Response.Usage.Input)
	}
}
