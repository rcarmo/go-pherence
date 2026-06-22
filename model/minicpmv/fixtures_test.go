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

func TestLoadMiniCPMOFixtureMetadata(t *testing.T) {
	meta, err := LoadMiniCPMOFixtureMetadata()
	if err != nil {
		t.Fatalf("LoadMiniCPMOFixtureMetadata: %v", err)
	}
	if meta.Summary.ModelType != "minicpm-o" || meta.AudioSpecialTokenIDs == nil || !meta.ReadinessReport.MetadataReady {
		t.Fatalf("bad fixture metadata: %+v", meta)
	}
}
