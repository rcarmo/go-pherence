package minicpmv

import (
	"os"
	"path/filepath"
	"testing"
)

func TestMiniCPMOFixturePath(t *testing.T) {
	path := filepath.Join("..", "..", MiniCPMOFixturePath, "config.json")
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("fixture path %s missing: %v", path, err)
	}
}
