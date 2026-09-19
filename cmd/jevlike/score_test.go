package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestScoreAssetIdentityRejectsTamperingAndMissingShards(t *testing.T) {
	dir := t.TempDir()
	paths := []string{"config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"}
	manifest := verifiedAssets{Version: 1}
	manifest.Sources = append(manifest.Sources, struct {
		Repository string `json:"repository"`
		Revision   string `json:"revision"`
	}{"Qwen/Qwen3-4B-Base", strings.Repeat("a", 40)})
	for _, name := range paths {
		path := filepath.Join(dir, name)
		data := []byte(name)
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		h := sha256.Sum256(data)
		manifest.Files = append(manifest.Files, verifiedAsset{path, int64(len(data)), hex.EncodeToString(h[:])})
	}
	path := filepath.Join(dir, "verified.json")
	data, _ := json.Marshal(manifest)
	os.WriteFile(path, data, 0o600)
	first, err := verifyScoreAssets(dir, path)
	if err != nil || !strings.Contains(first, "#sha256=") {
		t.Fatal(first, err)
	}
	// Manifest ordering and absolute storage paths are not content identity.
	manifest.Files[0], manifest.Files[1] = manifest.Files[1], manifest.Files[0]
	data, _ = json.Marshal(manifest)
	os.WriteFile(path, data, 0o600)
	second, err := verifyScoreAssets(dir, path)
	if err != nil || first != second {
		t.Fatal("unstable identity", second, err)
	}
	manifest.Sources[0].Revision = strings.Repeat("z", 40)
	data, _ = json.Marshal(manifest)
	os.WriteFile(path, data, 0o600)
	if _, err = verifyScoreAssets(dir, path); err == nil {
		t.Fatal("nonhex revision accepted")
	}
	manifest.Sources[0].Revision = strings.Repeat("a", 40)
	data, _ = json.Marshal(manifest)
	os.WriteFile(path, data, 0o600)
	os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("tampered"), 0o600)
	if _, err := verifyScoreAssets(dir, path); err == nil {
		t.Fatal("accepted different encoder bytes")
	}
	os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("model.safetensors"), 0o600)
	os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(`{"weight_map":{"x":"outside.safetensors"}}`), 0o600)
	if _, err := verifyScoreAssets(dir, path); err == nil {
		t.Fatal("unverified index overrides single weights")
	}
}
func TestScoreHelpAndMissingArguments(t *testing.T) {
	var out, errout bytes.Buffer
	if err := run([]string{"score", "-h"}, &out, &errout); err != nil {
		t.Fatal(err)
	}
	if err := run([]string{"score"}, &out, &errout); err == nil {
		t.Fatal("missing paths accepted")
	}
}
