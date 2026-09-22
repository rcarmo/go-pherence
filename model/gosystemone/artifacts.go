package gosystemone

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// VerifyV1Artifacts validates the exact model and tokenizer files frozen by the
// Go System One v1 provenance contract. Verification happens before either artifact is
// parsed so a server cannot accidentally start with a similarly named export.
func VerifyV1Artifacts(modelPath, tokenizerDir string) error {
	if err := verifyArtifact(modelPath, V1Provenance.ModelBytes, V1Provenance.ModelSHA256); err != nil {
		return fmt.Errorf("model: %w", err)
	}
	for name, want := range map[string]string{
		"tokenizer.json":        V1Provenance.TokenizerSHA256,
		"tokenizer_config.json": V1Provenance.TokenizerConfigSHA256,
		"chat_template.jinja":   V1Provenance.ChatTemplateSHA256,
	} {
		if err := verifyArtifact(filepath.Join(tokenizerDir, name), -1, want); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
	}
	return nil
}

func verifyArtifact(path string, wantBytes int64, wantSHA256 string) error {
	if path == "" {
		return fmt.Errorf("empty path")
	}
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("not a regular file")
	}
	if wantBytes >= 0 && info.Size() != wantBytes {
		return fmt.Errorf("bytes=%d, want %d", info.Size(), wantBytes)
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != wantSHA256 {
		return fmt.Errorf("SHA-256=%s, want %s", got, wantSHA256)
	}
	return nil
}
