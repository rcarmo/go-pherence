package silero

import (
	"os"
	"path/filepath"
	"testing"
)

func TestBoundedReferenceFixtureIO(t *testing.T) {
	path := filepath.Join(t.TempDir(), "reference")
	if err := os.WriteFile(path, []byte("small fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	got, err := boundedReferenceFile(path, 32)
	if err != nil || string(got) != "small fixture" {
		t.Fatal(string(got), err)
	}
	if got, err := boundedReferenceFile(path, 3); err == nil || got != nil {
		t.Fatal("oversized oracle fixture", got, err)
	}
	if _, err := boundedReferenceFile(path+"-missing", 32); err == nil {
		t.Fatal("missing oracle fixture")
	}
}
