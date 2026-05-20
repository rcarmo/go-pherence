package boundary

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestBERTDoesNotImportGonumBLAS(t *testing.T) {
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	err = filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() || filepath.Ext(path) != ".go" || filepath.Base(path) == "import_boundary_test.go" {
			return err
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if bytes.Contains(data, []byte("gonum.org/v1/gonum/blas")) {
			rel, _ := filepath.Rel(dir, path)
			t.Fatalf("%s imports Gonum BLAS; use checked SIMD runtime APIs", rel)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk BERT package: %v", err)
	}
}
