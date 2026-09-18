package gguf

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func TestMapReadOnlyExactData(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mapped.bin")
	want := []byte("hello, gguf mmap\x00with bytes")
	if err := os.WriteFile(path, want, 0o644); err != nil {
		t.Fatalf("write temp file: %v", err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open temp file: %v", err)
	}
	g := &GGUF{f: f}
	defer g.Close()

	got, release, err := g.MapReadOnly()
	if err != nil {
		t.Fatalf("MapReadOnly: %v", err)
	}
	defer func() {
		if err := release(); err != nil {
			t.Fatalf("release: %v", err)
		}
	}()

	if !bytes.Equal(got, want) {
		t.Fatalf("mapped bytes mismatch\n got=%v\nwant=%v", got, want)
	}
}

func TestMapReadOnlySurvivesFileClose(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mapped.bin")
	want := []byte("mapping survives file close")
	if err := os.WriteFile(path, want, 0o644); err != nil {
		t.Fatalf("write temp file: %v", err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open temp file: %v", err)
	}
	g := &GGUF{f: f}

	got, release, err := g.MapReadOnly()
	if err != nil {
		g.Close()
		t.Fatalf("MapReadOnly: %v", err)
	}
	defer func() {
		if err := release(); err != nil {
			t.Fatalf("release: %v", err)
		}
	}()
	g.Close()

	if !bytes.Equal(got, want) {
		t.Fatalf("mapped bytes after close mismatch\n got=%v\nwant=%v", got, want)
	}
	if got[0] != want[0] || got[len(got)-1] != want[len(want)-1] {
		t.Fatalf("mapped slice was not readable after Close")
	}
}

func TestMapReadOnlyRejectsInvalidSize(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.bin")
	if err := os.WriteFile(path, nil, 0o644); err != nil {
		t.Fatalf("write temp file: %v", err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open temp file: %v", err)
	}
	g := &GGUF{f: f}
	defer g.Close()

	if _, _, err := g.MapReadOnly(); err == nil || !strings.Contains(err.Error(), "size") {
		t.Fatalf("MapReadOnly empty err=%v", err)
	}
}

func TestMapReadOnlyReleaseOnceContractCanBeWrappedIdempotent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mapped.bin")
	if err := os.WriteFile(path, []byte("release once"), 0o644); err != nil {
		t.Fatalf("write temp file: %v", err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open temp file: %v", err)
	}
	g := &GGUF{f: f}
	defer g.Close()

	_, release, err := g.MapReadOnly()
	if err != nil {
		t.Fatalf("MapReadOnly: %v", err)
	}

	var once sync.Once
	var wrappedErr error
	wrappedRelease := func() error {
		once.Do(func() {
			wrappedErr = release()
		})
		return wrappedErr
	}

	if err := wrappedRelease(); err != nil {
		t.Fatalf("wrapped release first call: %v", err)
	}
	if err := wrappedRelease(); err != nil {
		t.Fatalf("wrapped release second call: %v", err)
	}
}
