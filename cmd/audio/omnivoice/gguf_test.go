package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestGGUFCLIExport(t *testing.T) {
	old := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(old)
	model := filepath.Join("..", "..", "..", "testdata", "omnivoice", "backbone")
	out := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := run([]string{"-mode", "export-gguf", "-model", model, "-output", out, "-gguf-format", "f32", "-threads", "1"}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(out); err != nil {
		t.Fatal(err)
	}
	if err := run([]string{"-mode", "export-gguf", "-model", model, "-output", out, "-threads", "1"}); err == nil {
		t.Fatal("overwrote file")
	}
	if err := run([]string{"-mode", "prepare", "-weights-gguf", out, "-threads", "1"}); err == nil || !strings.Contains(err.Error(), "weights-gguf applies") {
		t.Fatalf("%v", err)
	}
}

func TestDirectQ8FlagValidation(t *testing.T) {
	for _, args := range [][]string{
		{"-mode", "generate", "-direct-q8"},
		{"-mode", "generate", "-direct-q8", "-weights-gguf", "x.gguf", "-resident-mib", "1700"},
		{"-mode", "logits", "-direct-q8", "-weights-gguf", "x.gguf"},
	} {
		if err := run(args); err == nil || !strings.Contains(err.Error(), "direct-q8 requires") {
			t.Fatalf("args %v: %v", args, err)
		}
	}
}
