package gosystemone

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestVerifyArtifact(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fixture")
	body := []byte("go-system-one fixture\n")
	if err := os.WriteFile(path, body, 0o600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(body)
	sha := hex.EncodeToString(sum[:])
	if err := verifyArtifact(path, int64(len(body)), sha); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		path string
		size int64
		sha  string
		want string
	}{
		{"empty", "", -1, sha, "empty path"},
		{"missing", path + ".missing", -1, sha, "no such file"},
		{"size", path, int64(len(body)) + 1, sha, "bytes="},
		{"hash", path, -1, strings.Repeat("0", 64), "SHA-256="},
		{"directory", t.TempDir(), -1, sha, "not a regular file"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := verifyArtifact(tc.path, tc.size, tc.sha)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error=%v want substring %q", err, tc.want)
			}
		})
	}
}
