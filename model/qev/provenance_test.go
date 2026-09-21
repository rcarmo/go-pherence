package qev

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestV1ProvenancePinsExactArtifactsAndSources(t *testing.T) {
	p := V1Provenance
	for name, value := range map[string]string{
		"model repository": p.ModelRepository, "model revision": p.ModelRevision,
		"model file": p.ModelFile, "model hash": p.ModelSHA256,
		"tokenizer repository": p.TokenizerRepository, "tokenizer revision": p.TokenizerRevision,
		"tokenizer hash": p.TokenizerSHA256, "tokenizer config hash": p.TokenizerConfigSHA256,
		"chat template hash": p.ChatTemplateSHA256, "llama revision": p.LlamaRevision,
		"RLCD revision": p.RLCDRevision, "playground revision": p.PlaygroundRevision,
	} {
		if value == "" {
			t.Fatalf("empty provenance %s", name)
		}
	}
	if p.ModelBytes != 7366423360 || p.ModelLicense != "Apache-2.0" || p.LlamaLicense != "MIT" || p.PlaygroundLicense != "none; behavior reference only" {
		t.Fatalf("provenance=%+v", p)
	}
}

func TestV1ProvenanceMatchesCheckedInManifest(t *testing.T) {
	data, err := os.ReadFile("testdata/provenance.json")
	if err != nil {
		t.Fatal(err)
	}
	var manifest struct {
		Schema string `json:"schema"`
		Model  struct {
			Repository, Revision, File, SHA256, License string
			Bytes                                       int64
		}
		Tokenizer struct {
			Repository, Revision string
			Files                map[string]string
		}
		Runtime struct {
			DecisionSequences int `json:"decision_sequences"`
			ContextTokens     int `json:"context_tokens"`
			UBatchTokens      int `json:"ubatch_tokens"`
			MaxContexts       int `json:"max_contexts"`
			MaxFields         int `json:"max_fields"`
			MaxCandidates     int `json:"max_candidates_per_field"`
		} `json:"runtime_bounds"`
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	p := V1Provenance
	if manifest.Schema != "go-pherence-qev-provenance-v1" || manifest.Model.Repository != p.ModelRepository || manifest.Model.Revision != p.ModelRevision || manifest.Model.File != p.ModelFile || manifest.Model.SHA256 != p.ModelSHA256 || manifest.Model.Bytes != p.ModelBytes || manifest.Model.License != p.ModelLicense {
		t.Fatalf("model manifest=%+v provenance=%+v", manifest.Model, p)
	}
	if manifest.Tokenizer.Repository != p.TokenizerRepository || manifest.Tokenizer.Revision != p.TokenizerRevision || manifest.Tokenizer.Files["tokenizer.json"] != p.TokenizerSHA256 || manifest.Tokenizer.Files["tokenizer_config.json"] != p.TokenizerConfigSHA256 || manifest.Tokenizer.Files["chat_template.jinja"] != p.ChatTemplateSHA256 {
		t.Fatalf("tokenizer manifest=%+v provenance=%+v", manifest.Tokenizer, p)
	}
	if manifest.Runtime.DecisionSequences != DecisionSequences || manifest.Runtime.ContextTokens != MaxContextTokens || manifest.Runtime.UBatchTokens != UBatchTokens || manifest.Runtime.MaxContexts != MaxContexts || manifest.Runtime.MaxFields != MaxFields || manifest.Runtime.MaxCandidates != MaxCandidates {
		t.Fatalf("runtime manifest=%+v", manifest.Runtime)
	}
}

func TestPinnedGemma4Artifacts(t *testing.T) {
	modelPath := os.Getenv("GO_PHERENCE_GEMMA4_12B")
	tokenizerDir := os.Getenv("GO_PHERENCE_GEMMA4_12B_TOKENIZER")
	if modelPath == "" || tokenizerDir == "" {
		t.Skip("set GO_PHERENCE_GEMMA4_12B and GO_PHERENCE_GEMMA4_12B_TOKENIZER for pinned artifact checks")
	}
	assertPinnedFile(t, modelPath, V1Provenance.ModelBytes, V1Provenance.ModelSHA256)
	for name, want := range map[string]string{
		"tokenizer.json":        V1Provenance.TokenizerSHA256,
		"tokenizer_config.json": V1Provenance.TokenizerConfigSHA256,
		"chat_template.jinja":   V1Provenance.ChatTemplateSHA256,
	} {
		assertPinnedFile(t, filepath.Join(tokenizerDir, name), -1, want)
	}
}

func assertPinnedFile(t *testing.T, path string, wantSize int64, wantSHA string) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	if wantSize >= 0 && info.Size() != wantSize {
		t.Fatalf("%s bytes=%d want %d", path, info.Size(), wantSize)
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		t.Fatal(err)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != wantSHA {
		t.Fatalf("%s SHA-256=%s want %s", path, got, wantSHA)
	}
}
