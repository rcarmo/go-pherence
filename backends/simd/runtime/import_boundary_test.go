package simd

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestSIMDKernelImportBoundary(t *testing.T) {
	repo := findSIMDRepoRoot(t)
	for _, root := range []string{"model", "models", "tensor", "runtime", filepath.Join("backends", "nvidia"), filepath.Join("backends", "vulkan"), filepath.Join("backends", "mlx")} {
		rootPath := filepath.Join(repo, root)
		if _, err := os.Stat(rootPath); err != nil {
			continue
		}
		err := filepath.WalkDir(rootPath, func(path string, d os.DirEntry, err error) error {
			if err != nil || d.IsDir() || filepath.Ext(path) != ".go" {
				return err
			}
			base := filepath.Base(path)
			if base == "vulkan_more_parity_test.go" || base == "import_boundary_test.go" {
				return nil
			}
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			if bytes.Contains(data, []byte("github.com/rcarmo/go-pherence/backends/simd/kernels")) {
				rel, _ := filepath.Rel(repo, path)
				t.Fatalf("%s imports SIMD kernels directly; use backends/simd/runtime", rel)
			}
			return nil
		})
		if err != nil {
			t.Fatalf("walk %s: %v", root, err)
		}
	}
}

func TestUnsafeSGEMMCallBoundary(t *testing.T) {
	repo := findSIMDRepoRoot(t)
	for _, root := range []string{"model", "models", "tensor", "runtime", filepath.Join("backends", "nvidia"), filepath.Join("backends", "vulkan"), filepath.Join("backends", "mlx")} {
		rootPath := filepath.Join(repo, root)
		if _, err := os.Stat(rootPath); err != nil {
			continue
		}
		err := filepath.WalkDir(rootPath, func(path string, d os.DirEntry, err error) error {
			if err != nil || d.IsDir() || filepath.Ext(path) != ".go" {
				return err
			}
			if filepath.Base(path) == "import_boundary_test.go" {
				return nil
			}
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			if bytes.Contains(data, []byte(".SgemmNN(")) || bytes.Contains(data, []byte(".SgemmNT(")) {
				rel, _ := filepath.Rel(repo, path)
				t.Fatalf("%s calls unsafe SIMD SGEMM directly; use SgemmNNTo/SgemmNTTo", rel)
			}
			return nil
		})
		if err != nil {
			t.Fatalf("walk %s: %v", root, err)
		}
	}
}

func findSIMDRepoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("repository root not found")
		}
		dir = parent
	}
}
