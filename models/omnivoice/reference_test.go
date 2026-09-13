package omnivoice

import (
	"context"
	"encoding/json"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestRealReferenceParity(t *testing.T) {
	root := os.Getenv("GO_PHERENCE_REAL_CODEC")
	if root == "" {
		t.Skip("set GO_PHERENCE_REAL_CODEC")
	}
	python := os.Getenv("GO_PHERENCE_REFERENCE_PYTHON")
	if python == "" {
		t.Skip("set GO_PHERENCE_REFERENCE_PYTHON")
	}
	path := filepath.Join(t.TempDir(), "reference.json")
	cmd := exec.Command(python, "../../scripts/omnivoice-reference-fixture.py", "--model", root, "--output", path)
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("fixture: %v %s", err, b)
	}
	var f struct {
		Wave, Resampled []float32
		Reference       loader.CachedReferenceTokens
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	normalized := append([]float32(nil), f.Wave...)
	gain := float32(.1 / *f.Reference.RefRMS)
	for i := range normalized {
		normalized[i] *= gain
	}
	r := resample24To16(normalized)
	var peak float64
	for i := range r {
		peak = max(peak, math.Abs(float64(r[i]-f.Resampled[i])))
	}
	if peak > 1e-6 {
		t.Fatalf("resample max error %g", peak)
	}
	e, err := LoadReferenceEncoder(root)
	if err != nil {
		t.Fatal(err)
	}
	got, err := e.Encode(context.Background(), f.Wave, f.Reference.Transcript)
	if err != nil {
		t.Fatal(err)
	}
	if got.Frames != f.Reference.Frames || len(got.Codes) != len(f.Reference.Codes) {
		t.Fatal("shape mismatch")
	}
	mismatch := 0
	for i := range got.Codes {
		if got.Codes[i] != f.Reference.Codes[i] {
			mismatch++
		}
	}
	t.Logf("frames=%d code mismatches=%d/%d resample maxerr=%g rms=%g/%g", got.Frames, mismatch, len(got.Codes), peak, *got.RefRMS, *f.Reference.RefRMS)
	if mismatch != 0 {
		t.Fatal("reference code parity failed")
	}
}
