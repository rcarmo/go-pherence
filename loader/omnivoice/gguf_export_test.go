package omnivoice

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestExportGGUFRoundTrip(t *testing.T) {
	source := filepath.Join("..", "..", "testdata", "omnivoice", "backbone")
	w, err := OpenWeights(source)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	for _, format := range []string{"f16", "f32"} {
		t.Run(format, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "model.gguf")
			if err := ExportGGUF(context.Background(), w, path, format); err != nil {
				t.Fatal(err)
			}
			cfg, err := LoadConfig(path)
			if err != nil || !reflect.DeepEqual(cfg, w.Config) {
				t.Fatalf("config %v", err)
			}
			got, err := OpenWeights(path)
			if err != nil {
				t.Fatal(err)
			}
			defer got.Close()
			for _, name := range w.file.Names() {
				if name == "codebook_layer_offsets" {
					continue
				}
				a, sa, err := w.file.GetFloat32(name)
				if err != nil {
					t.Fatal(err)
				}
				b, sb, err := got.file.GetFloat32(name)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(sa, sb) {
					t.Fatal("shape")
				}
				for i := range a {
					want := a[i]
					if format == "f16" {
						want = half.F16ToF32(half.F32ToF16(want))
					}
					if want != b[i] {
						t.Fatalf("%s[%d] got %g want %g", name, i, b[i], want)
					}
				}
			}
			before, _ := os.ReadFile(path)
			if err := ExportGGUF(context.Background(), w, path, format); err == nil {
				t.Fatal("overwrote export")
			}
			after, _ := os.ReadFile(path)
			if !bytes.Equal(before, after) {
				t.Fatal("changed destination")
			}
		})
	}
}
func TestExportGGUFRejectsAndCancels(t *testing.T) {
	w, err := OpenWeights(filepath.Join("..", "..", "testdata", "omnivoice", "backbone"))
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := ExportGGUF(context.Background(), w, path, "q8"); err == nil {
		t.Fatal("format")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := ExportGGUF(ctx, w, path, "f32"); err == nil {
		t.Fatal("cancel")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("partial file")
	}
}
