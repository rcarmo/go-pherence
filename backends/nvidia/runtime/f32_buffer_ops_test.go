package nvidia

import "testing"

func TestVecScaleF32BufferPTXParameterOrder(t *testing.T) {
	if !SgemmReady() {
		if Available() {
			t.Fatal("CUDA device available but PTX runtime not ready")
		}
		t.Skip("CUDA unavailable")
	}
	buf, err := Malloc(4)
	if err != nil {
		t.Fatal(err)
	}
	defer buf.Free()
	if err := buf.Upload([]float32{11, -22, 33, -44}); err != nil {
		t.Fatal(err)
	}
	if err := VecScaleF32Buffer(buf, buf, 4, 0.5); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, 4)
	if err := buf.Download(got); err != nil {
		t.Fatal(err)
	}
	want := []float32{5.5, -11, 16.5, -22}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%g want %g", i, got[i], want[i])
		}
	}
}
