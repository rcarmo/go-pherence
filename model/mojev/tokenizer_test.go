package mojev

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestReleasedMoJevTokenizer(t *testing.T) {
	for _, pin := range []struct{ path, hash string }{
		{"../../scripts/mojev_oracle_tokenizer.py", "626859ba9470fe2c6ac92d6c6b7835fe2337c0fe47a3df8af7e0669c56f6924d"},
		{"testdata/released_tokenizer.json", "906157ed4ea409843e51c34fa162b924a7cea8aba927deae5d6f317d80c49818"},
	} {
		data, err := os.ReadFile(pin.path)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(data)) != pin.hash {
			t.Fatalf("oracle hash changed: %s", pin.path)
		}
	}
	data, err := os.ReadFile("testdata/released_tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema         int    `json:"schema"`
		Revision       string `json:"model_revision"`
		TokenizerSHA   string `json:"tokenizer_sha256"`
		ConfigSHA      string `json:"tokenizer_config_sha256"`
		TokenizerClass string `json:"tokenizer_class"`
		PadID          int    `json:"pad_id"`
		EOSID          int    `json:"eos_id"`
		Cases          []struct {
			Text string `json:"text"`
			IDs  []int  `json:"ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != 1 || fixture.Revision != ModelRevision || fixture.TokenizerSHA != "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523" || fixture.ConfigSHA != "66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b" || fixture.TokenizerClass != "Qwen2Tokenizer" || fixture.PadID != 248044 || fixture.EOSID != 248046 || len(fixture.Cases) != 18 {
		t.Fatal("unexpected tokenizer provenance")
	}
	path := os.Getenv("GO_PHERENCE_MOJEV_CHECKPOINT_DIR")
	if path == "" {
		t.Skip("set GO_PHERENCE_MOJEV_CHECKPOINT_DIR to pinned approved tokenizer assets")
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
	for _, tc := range fixture.Cases {
		t.Run(fmt.Sprintf("%q", tc.Text), func(t *testing.T) {
			got := tok.Encode(tc.Text)
			if !reflect.DeepEqual(got, tc.IDs) {
				t.Fatalf("tokens for %q: got %v want %v", tc.Text, got, tc.IDs)
			}
		})
	}
}
