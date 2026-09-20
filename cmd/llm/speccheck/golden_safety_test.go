package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestGoldenRejectsTrailingJSONAndOversize(t *testing.T) {
	p := filepath.Join(t.TempDir(), "golden.json")
	for _, data := range [][]byte{[]byte(`{} {}`), []byte(`{} garbage`), make([]byte, (16<<20)+1)} {
		if err := os.WriteFile(p, data, 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := loadGolden(p); err == nil {
			t.Fatal("malformed golden accepted")
		}
	}
	if err := os.WriteFile(p, []byte("{}\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadGolden(p); err != nil {
		t.Fatal(err)
	}
}
