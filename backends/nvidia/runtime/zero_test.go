package nvidia

import "testing"

func TestZeroFloat32Buffer(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA not available")
	}
	buf, err := Malloc(8)
	if err != nil {
		t.Fatal(err)
	}
	defer buf.Free()
	if err := buf.Upload([]float32{1, 2, 3, 4, 5, 6, 7, 8}); err != nil {
		t.Fatal(err)
	}
	if err := ZeroFloat32Buffer(buf, 8); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, 8)
	if err := buf.Download(got); err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if v != 0 {
			t.Fatalf("i=%d got=%g want 0", i, v)
		}
	}
}
