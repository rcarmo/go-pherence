package model

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestLoadGGUFMoEExpertMatricesMissingRouter(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.gguf")
	if err := os.WriteFile(path, minimalGGUF(t), 0o644); err != nil {
		t.Fatal(err)
	}
	g, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	if _, _, _, _, err := loadGGUFMoEExpertMatrices(g, 0); err == nil || !strings.Contains(err.Error(), `"blk.0.ffn_gate_inp.weight" not found`) {
		t.Fatalf("expected missing router tensor error, got %v", err)
	}
}

func minimalGGUF(t *testing.T) []byte {
	t.Helper()
	// Empty tensor inventory still needs a data offset within the file.
	// Pad the 24-byte header to GGUF's default 32-byte alignment; the loader
	// must reject missing router tensors, not a truncated fixture header.
	b := make([]byte, 32)
	copy(b, []byte{'G', 'G', 'U', 'F', 3})
	return b
}
