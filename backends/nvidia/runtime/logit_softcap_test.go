package nvidia

import (
	"math"
	"testing"
)

func TestLogitSoftcapF32(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	in := []float32{-60, -30, 0, 30, 60}
	buf, err := Malloc(len(in))
	if err != nil {
		t.Fatal(err)
	}
	defer buf.Free()
	if err := buf.Upload(in); err != nil {
		t.Fatal(err)
	}
	if err := LogitSoftcapF32(buf, len(in), 30); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, len(in))
	if err := buf.Download(got); err != nil {
		t.Fatal(err)
	}
	for i, v := range in {
		want := float32(math.Tanh(float64(v/30)) * 30)
		if math.Abs(float64(got[i]-want)) > 5e-3 {
			t.Fatalf("softcap[%d]=%g want %g all=%v", i, got[i], want, got)
		}
	}
}
