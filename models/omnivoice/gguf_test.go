package omnivoice

import (
	"context"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	"path/filepath"
	"testing"
)

func TestGGUFBackboneF32ExactAndZeroAlloc(t *testing.T) {
	original, f := loadBackboneFixture(t)
	path := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := loader.ExportGGUF(context.Background(), original.weights, path, "f32"); err != nil {
		t.Fatal(err)
	}
	w, err := loader.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	b, err := NewBackbone(w, f.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	a := make([]float32, len(f.Logits))
	out := make([]float32, len(f.Logits))
	ctx := context.Background()
	if err := original.ForwardInto(ctx, a, f.IDs, f.AudioMask, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := b.ForwardInto(ctx, out, f.IDs, f.AudioMask, nil, nil); err != nil {
		t.Fatal(err)
	}
	for i := range a {
		if a[i] != out[i] {
			t.Fatalf("logit %d: %g != %g", i, a[i], out[i])
		}
	}
	if n := testing.AllocsPerRun(10, func() {
		if err := b.ForwardInto(ctx, out, f.IDs, f.AudioMask, nil, nil); err != nil {
			panic(err)
		}
	}); n != 0 {
		t.Fatalf("allocations %g", n)
	}
}
