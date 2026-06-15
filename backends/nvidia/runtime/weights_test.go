package nvidia

import "testing"

func TestMulWeights(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	a, err := Malloc(3)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Free()
	b, err := Malloc(3)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Free()
	out, err := Malloc(3)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Free()
	if err := a.Upload([]float32{2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	if err := b.Upload([]float32{0.5, 0.25, 2}); err != nil {
		t.Fatal(err)
	}
	if err := MulWeights(out, a, b, 3); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, 3)
	if err := out.Download(got); err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 0.75, 8}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got=%v want=%v", got, want)
		}
	}
}
