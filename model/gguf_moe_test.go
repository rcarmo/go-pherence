package model

import (
	"os"
	"path/filepath"
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
	if _, _, _, _, err := loadGGUFMoEExpertMatrices(g, 0); err == nil {
		t.Fatal("expected missing router tensor error")
	}
}

func minimalGGUF(t *testing.T) []byte {
	t.Helper()
	// magic + version(3) + n_tensors(0) + n_kv(0) + alignment padding-free data section
	return []byte{'G', 'G', 'U', 'F', 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
}
