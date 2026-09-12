package whisper

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// Optional public tokenizer/config qualification. No weights, audio, GPU or
// neural inference. Download paths/revisions/checksums are in the manifest.
func TestCheckedTimestampPinnedTokenizers(t *testing.T) {
	root := os.Getenv("GO_PHERENCE_TEST_TOKENIZER_DIR")
	if root == "" {
		t.Skip("set GO_PHERENCE_TEST_TOKENIZER_DIR to pinned public tokenizer/config files")
	}
	var manifest struct {
		Tokenizers []struct {
			Model         string `json:"model"`
			TokenizerSHA  string `json:"tokenizer_sha256"`
			GenerationSHA string `json:"generation_sha256"`
			ConfigSHA     string `json:"config_sha256"`
		} `json:"tokenizers"`
	}
	data, err := os.ReadFile("../../docs/speech-generation-manifest.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	if len(manifest.Tokenizers) != 2 {
		t.Fatal("expected two pinned vocabularies")
	}
	for _, item := range manifest.Tokenizers {
		t.Run(item.Model, func(t *testing.T) {
			read := func(kind, want string) []byte {
				t.Helper()
				data, err := os.ReadFile(filepath.Join(root, item.Model+"-"+kind+".json"))
				if err != nil {
					t.Fatal(err)
				}
				sum := sha256.Sum256(data)
				if hex.EncodeToString(sum[:]) != want {
					t.Fatalf("%s checksum mismatch", kind)
				}
				return data
			}
			_ = read("tokenizer", item.TokenizerSHA)
			genData := read("generation", item.GenerationSHA)
			cfg := Tiny()
			if item.Model == "whisper-large-v3-turbo" {
				cfg = LargeV3Turbo()
			}
			tok, err := LoadTokenizer(filepath.Join(root, item.Model+"-tokenizer.json"))
			if err != nil {
				t.Fatal(err)
			}
			parsed, err := ParseModelConfigChecked(read("config", item.ConfigSHA))
			if err != nil || parsed != cfg {
				t.Fatalf("model config mismatch: %+v %v", parsed, err)
			}
			policy, err := ParseGenerationConfigChecked(genData, parsed, tok)
			if err != nil {
				t.Fatal(err)
			}
			if policy.maxLength != 448 || policy.maxInitial != 50 || len(policy.beginSuppress) != 2 || policy.beginSuppress[1] != 50257 {
				t.Fatal("generation defaults not imported")
			}
			v, err := checkedTimestampVocabulary(cfg, tok, "pt")
			if err != nil {
				t.Fatal(err)
			}
			var gen struct {
				SOT          int            `json:"decoder_start_token_id"`
				EOT          int            `json:"eos_token_id"`
				NoTimestamps int            `json:"no_timestamps_token_id"`
				Tasks        map[string]int `json:"task_to_id"`
				Languages    map[string]int `json:"lang_to_id"`
				MaxLength    int            `json:"max_length"`
				MaxInitial   int            `json:"max_initial_timestamp_index"`
			}
			if err := json.Unmarshal(genData, &gen); err != nil {
				t.Fatal(err)
			}
			if v.sot != gen.SOT || v.eot != gen.EOT || v.noTimestamps != gen.NoTimestamps || v.transcribe != gen.Tasks["transcribe"] || v.language != gen.Languages["<|pt|>"] || cfg.MaxDecoderLength != gen.MaxLength || gen.MaxInitial != 50 {
				t.Fatal("generation-config/tokenizer contract mismatch")
			}
		})
	}
}
