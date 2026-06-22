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

func TestLoadMiniCPMOFixtureExpectedSummary(t *testing.T) {
	s, err := LoadMiniCPMOFixtureExpectedSummary()
	if err != nil {
		t.Fatalf("LoadMiniCPMOFixtureExpectedSummary: %v", err)
	}
	if s.ModelType != "minicpm-o" || s.AudioPatchTokenID != 151653 || s.RuntimeReady {
		t.Fatalf("bad expected summary: %+v", s)
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
