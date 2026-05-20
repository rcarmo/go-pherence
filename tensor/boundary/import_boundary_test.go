package boundary

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestTensorUsesCheckedSIMDRuntimeBoundaries(t *testing.T) {
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
		if bytes.Contains(data, []byte("github.com/rcarmo/go-pherence/backends/simd/kernels")) {
			rel, _ := filepath.Rel(dir, path)
			t.Fatalf("%s imports SIMD kernels directly; use checked runtime APIs", rel)
		}
		if bytes.Contains(data, []byte(".SgemmNN(")) || bytes.Contains(data, []byte(".SgemmNT(")) {
			rel, _ := filepath.Rel(dir, path)
			t.Fatalf("%s calls unsafe SIMD SGEMM directly; use checked runtime APIs", rel)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk tensor package: %v", err)
	}
}
