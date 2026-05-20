package quant

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBackendCodeDoesNotImportRuntimeQuant(t *testing.T) {
	repo := findRepoRoot(t)
	for _, root := range []string{"backends", "model", "models", "tensor"} {
		rootPath := filepath.Join(repo, root)
		if _, err := os.Stat(rootPath); err != nil {
			continue
		}
		err := filepath.WalkDir(rootPath, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				return nil
			}
			if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			if bytes.Contains(data, []byte("github.com/rcarmo/go-pherence/runtime/quant")) {
				rel, _ := filepath.Rel(repo, path)
				t.Fatalf("%s imports runtime/quant; backend-owned code must import the owning backend package directly", rel)
			}
			return nil
		})
		if err != nil {
			t.Fatalf("walk %s: %v", root, err)
		}
	}
}

func TestRuntimeQuantContainsOnlyCompatibilityWrappers(t *testing.T) {
	repo := findRepoRoot(t)
	root := filepath.Join(repo, "runtime", "quant")
	allowedFiles := map[string]bool{
		"gemv_q4.go":          true,
		"gemv_q4_validate.go": true,
		"gptq.go":             true,
		"gptq_validate.go":    true,
		"mlx.go":              true,
		"nvfp4.go":            true,
	}
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		name := filepath.Base(path)
		if !allowedFiles[name] {
			t.Fatalf("unexpected runtime/quant implementation file %s; add backend code under its owning backend package", name)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk runtime/quant: %v", err)
	}
}

func findRepoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		next := filepath.Dir(dir)
		if next == dir {
			t.Fatal("go.mod not found")
		}
		dir = next
	}
}
